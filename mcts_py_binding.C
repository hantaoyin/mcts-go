// -*- mode:c++; c-basic-offset:2 -*-
#include <iostream>
#include <type_traits>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "mcts.h"

namespace {
class PyEvalWrapper {
  static constexpr size_t BoardSize = go_engine::N * go_engine::N;
 public:
  PyEvalWrapper(PyObject* _eval)
    : eval(_eval)
  {
    CHECK(PyCallable_Check(eval)) << "Python object is not callable: " << PyUnicode_AsASCIIString(PyObject_Str(eval));
    Py_XINCREF(eval);
    npy_intp dims[4] = {1, 3, go_engine::N, go_engine::N};
    args = PyTuple_New(1);
    PyObject* array_obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT, input.data());
    PyTuple_SetItem(args, 0, array_obj);
  }
  
  PyEvalWrapper(PyEvalWrapper&& other) = default;
  // Be careful about proper ref counting and pointers inside PyTuple if copy constructor is needed. 
  PyEvalWrapper(const PyEvalWrapper& other) = delete;
  const PyEvalWrapper& operator=(const PyEvalWrapper&) = delete;

  ~PyEvalWrapper() {
    Py_XDECREF(args);
    Py_XDECREF(eval);
  }

  float operator()(const go_engine::BoardInfo& b, go_engine::Color c, std::array<float, mcts::TotalMoves>& prior) {
    for (size_t m = 0; m < BoardSize; ++m) {
      input[m] = b.has_stone(m, c);
      input[m + BoardSize] = b.has_stone(m, go_engine::opposite_color(c));
      input[m + 2 * BoardSize] = c;
    }

    PyObject* result = PyObject_CallObject(eval, args);
    if (result == nullptr) {
      PyErr_PrintEx(1);
      CHECK(false) << "nullptr returned.";
    }

    CHECK(PyTuple_Check(result));
    CHECK(PyTuple_Size(result) == 2) << "Callback returns a tuple of size " << PyTuple_Size(result);

    PyObject* policy_output = PyTuple_GetItem(result, 0);
    PyObject* value_output = PyTuple_GetItem(result, 1);
    CHECK(PyArray_Check(policy_output)) << "Return value 1 is not PyArray.";
    CHECK(PyFloat_Check(value_output)) << "Return value 2 is not PyFloat.";

    PyArrayObject* policy_array = (PyArrayObject*)policy_output;
    CHECK(PyArray_TYPE(policy_array) == NPY_FLOAT) << "Elements in returned PyArray are type " << PyArray_TYPE(policy_array) << ", expecting " << NPY_FLOAT;
    CHECK(PyArray_NDIM(policy_array) == 1) << "Returned PyArray has a dimension other than 1: " << PyArray_NDIM(policy_array);
    npy_intp* dims = PyArray_DIMS(policy_array);
    CHECK(dims[0] == go_engine::TotalMoves) << "Returned PyArray has size: " << dims[0] << ", expecting " << go_engine::TotalMoves;
    memcpy(prior.data(), PyArray_GETPTR1(policy_array, 0), sizeof(float) * go_engine::TotalMoves);
    float ret = PyFloat_AsDouble(value_output);
    Py_XDECREF(result);
    return ret;
  }
 private:
  std::array<float, 3 * BoardSize * BoardSize> input;
  PyObject* args;
  PyObject* eval;
};
}  // namespace

typedef struct {
  PyObject_HEAD
  mcts::Tree<PyEvalWrapper> tree;
} MCTObject;

namespace MCTPyBinding {
static PyObject* py_new(PyTypeObject* type, PyObject*, PyObject*) {
  MCTObject* self = (MCTObject*)(type->tp_alloc(type, 0));
  return (PyObject*)self;
}
static int py_init(MCTObject*, PyObject*, PyObject*);

static void dealloc(MCTObject* self) {
  self->tree.~Tree();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* reset(MCTObject* self) {
  self->tree.reset();
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* get_search_count(MCTObject* self) {
  // If they are not the same, change NPY_UINT appropriately.
  static_assert(std::is_same_v<unsigned, uint32_t>);
  const std::array<unsigned, go_engine::TotalMoves>& count(self->tree.get_search_count());
  npy_intp dims[1] = {go_engine::TotalMoves};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_UINT);
  memcpy(PyArray_GETPTR1(array, 0), count.data(), sizeof(uint32_t) * go_engine::TotalMoves);
  return array;
}

static PyObject* is_valid(MCTObject* self, PyObject* args) {
  int color, pos;
  if (!PyArg_ParseTuple(args, "ii", &color, &pos)) {
    return nullptr;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "1st arg (color) can only be 0 or 1.");
    return nullptr;
  }
  if (pos < 0 || pos >= (int)go_engine::TotalMoves) {
    PyErr_SetString(PyExc_ValueError, "2nd arg (position) can only be [0, N * N].");
    return nullptr;
  }
  go_engine::Move move((go_engine::Color)color, pos);
  if (self->tree.is_valid(move)) {
    Py_XINCREF(Py_True);
    return Py_True;
  } else {
    Py_XINCREF(Py_False);
    return Py_False;
  }
}

static PyObject* play(MCTObject* self, PyObject* args) {
  int color, pos;
  if (!PyArg_ParseTuple(args, "ii", &color, &pos)) {
    return nullptr;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "1st arg (color) can only be 0 or 1.");
    return nullptr;
  }
  if (pos < 0 || pos >= (int)go_engine::TotalMoves) {
    PyErr_SetString(PyExc_ValueError, "2nd arg (position) can only be [0, N * N].");
    return nullptr;
  }
  go_engine::Move move((go_engine::Color)color, pos);
  self->tree.play(move);
  Py_XINCREF(Py_None);
  return Py_None;
}

static PyObject* gen_play(MCTObject* self) {
  go_engine::Move move = self->tree.gen_play(true);
  return PyLong_FromUnsignedLong(move.id());
}

static PyObject* score(MCTObject* self) {
  double s = self->tree.score();
  return PyFloat_FromDouble(s);
}
};

static PyMethodDef MCT_methods[] = {
  {"reset", (PyCFunction)MCTPyBinding::reset, METH_NOARGS, "Reset the tree."},
  {"get_search_count", (PyCFunction)MCTPyBinding::get_search_count, METH_NOARGS, "Return the search / play out count of the current game state, this should always be called right after gen_play and before play."},
  {"is_valid", (PyCFunction)MCTPyBinding::is_valid, METH_VARARGS, "is_valid(color, pos): Test if a move is valid."},
  {"play", (PyCFunction)MCTPyBinding::play, METH_VARARGS, "play(color, pos): Play a move and change internal state."},
  {"gen_play", (PyCFunction)MCTPyBinding::gen_play, METH_NOARGS, "Gnerate a play using MCTS."},
  {"score", (PyCFunction)MCTPyBinding::score, METH_NOARGS, "Get my score - opponent's score."},
  {nullptr},
};

static PyTypeObject mct_py_type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "mcts.Tree",
  sizeof(MCTObject),  // tp_basicsize
  0,  // tp_itemsize
  (destructor)MCTPyBinding::dealloc,  // tp_dealloc
  0,  // tp_print
  0,  // tp_getattr
  0,  // tp_setattr
  0,  // tp_as_async
  0,  // tp_repr

  0,  // tp_as_number;
  0,  // tp_as_sequence
  0,  // tp_as_mapping

  0,  // tp_hash
  0,  // tp_call
  0,  // tp_str
  0,  // tp_getattro
  0,  // tp_setattro

  0,  // tp_as_buffer

  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
  "Monte Carlo search tree for game of Go.",  // tp_doc
  0,  // tp_traverse
  0,  // tp_clear
  0,  // tp_richcompare
  0,  // tp_weaklistoffset
  0,  // tp_iter
  0,  // tp_iternext

  MCT_methods,  // tp_methods
  0,  // tp_members
  0,  // tp_getset
  0,  // tp_base
  0,  // tp_dict
  0,  // tp_descr_get
  0,  // tp_descr_set
  0,  // tp_dictoffset
  (initproc)MCTPyBinding::py_init,  // tp_init
  0,  // tp_alloc
  MCTPyBinding::py_new,  // tp_new
  0,  // tp_free
  0,  // tp_is_gc
  0,  // tp_bases
  0,  // tp_mro
  0,  // tp_cache
  0,  // tp_subclasses
  0,  // tp_weaklist
  0,  // tp_del
  0,  // tp_version_tag
  0,  // tp_finalize
};

namespace MCTPyBinding {
static int py_init(MCTObject* self, PyObject* args, PyObject* kwargs) {
  char options_string[][10] = {"komi", "color", "eval"};
  char* kwlist[] = {options_string[0], options_string[1], options_string[2], nullptr};
  float komi;
  int color;
  PyObject* eval;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fiO", kwlist, &komi, &color, &eval)) {
    return -1;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "color can only be 0 or 1.");
    return -1;
  }
  new(&(self->tree)) mcts::Tree<PyEvalWrapper>(komi, (go_engine::Color)color, PyEvalWrapper(eval));
  return 0;
}
} // MCTPyBinding

static PyObject* board_size(PyObject*, PyObject*) {
  return PyLong_FromLong((long)go_engine::N);
}

static PyMethodDef module_methods[] = {
  {"board_size", board_size, METH_NOARGS, "Get board size."},
  {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef mcts_module = {
  PyModuleDef_HEAD_INIT,
  "mcts",
  nullptr,
  -1,
  module_methods,
};

PyMODINIT_FUNC PyInit_mcts(void) {
  if (PyType_Ready(&mct_py_type) < 0) {
    return nullptr;
  }
  PyObject* m = PyModule_Create(&mcts_module);
  Py_INCREF(&mct_py_type);
  PyModule_AddObject(m, "Tree", (PyObject*)&mct_py_type);
  import_array();
  return m;
}
