// -*- mode:c++; c-basic-offset:2 -*-
#include <iostream>
#include <type_traits>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "mcts.h"
#include "eval_bridge.h"

namespace EvalBridgePyBinding {
// This hard codes batch size as 32.
struct EvalBridgeObject {
  PyObject_HEAD
  mcts::NetworkEvalBridge<5> bridge;
};

static PyObject* py_new(PyTypeObject* type, PyObject*, PyObject*) {
  EvalBridgeObject* self = (EvalBridgeObject*)(type->tp_alloc(type, 0));
  return (PyObject*)self;
}
static int py_init(EvalBridgeObject* self, PyObject* args, PyObject* kwargs) {
  char options_string[][10] = {"eval"};
  char* kwlist[] = {options_string[0], nullptr};
  PyObject* eval;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &eval)) {
    return -1;
  }
  new(&(self->bridge)) mcts::NetworkEvalBridge<5>(eval);
  return 0;
}

static void dealloc(EvalBridgeObject* self) {
  self->bridge.~NetworkEvalBridge();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* worker_thread_count(EvalBridgeObject* self) {
  unsigned long count = self->bridge.worker_thread_count();
  return PyLong_FromUnsignedLong(count);
}

static PyObject* start_eval(EvalBridgeObject* self) {
  Py_BEGIN_ALLOW_THREADS
  self->bridge.startEval(_save);
  Py_END_ALLOW_THREADS
  Py_XINCREF(Py_None);
  return Py_None;
}
}  // namespace EvalBridgePyBinding

static PyMethodDef eval_bridge_methods[] = {
  {"worker_thread_count", (PyCFunction)EvalBridgePyBinding::worker_thread_count, METH_NOARGS, "Return the number of worker threads should be used with this eval object."},
  {"start_eval", (PyCFunction)EvalBridgePyBinding::start_eval, METH_NOARGS, "Start listening to eval requests, this function never returns."},
  {nullptr},
};

static PyTypeObject eval_bridge_py_type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "mcts.EvalBridge",
  sizeof(EvalBridgePyBinding::EvalBridgeObject),  // tp_basicsize
  0,  // tp_itemsize
  (destructor)EvalBridgePyBinding::dealloc,  // tp_dealloc
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
  "A class to group multiple eval requests from different threads into batches.",  // tp_doc
  0,  // tp_traverse
  0,  // tp_clear
  0,  // tp_richcompare
  0,  // tp_weaklistoffset
  0,  // tp_iter
  0,  // tp_iternext

  eval_bridge_methods,  // tp_methods
  0,  // tp_members
  0,  // tp_getset
  0,  // tp_base
  0,  // tp_dict
  0,  // tp_descr_get
  0,  // tp_descr_set
  0,  // tp_dictoffset
  (initproc)EvalBridgePyBinding::py_init,  // tp_init
  0,  // tp_alloc
  EvalBridgePyBinding::py_new,  // tp_new
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
struct MCTObject {
  PyObject_HEAD
  mcts::Tree<mcts::NetworkEvalBridge<5>&> tree;
};

static PyObject* py_new(PyTypeObject* type, PyObject*, PyObject*) {
  MCTObject* self = (MCTObject*)(type->tp_alloc(type, 0));
  return (PyObject*)self;
}

static int py_init(MCTObject* self, PyObject* args, PyObject* kwargs) {
  Py_BEGIN_ALLOW_THREADS
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
  if (PyObject_TypeCheck(eval, &eval_bridge_py_type)) {
    auto* obj = (EvalBridgePyBinding::EvalBridgeObject*)eval;
    new(&(self->tree)) mcts::Tree<mcts::NetworkEvalBridge<5>&>(komi, (go_engine::Color)color, obj->bridge);
  } else {
    PyErr_SetString(PyExc_ValueError, "Must pass a valid EvalBridge object.");
    return -1;
  }
  Py_END_ALLOW_THREADS
  return 0;
}

static void dealloc(MCTObject* self) {
  self->tree.~Tree();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* reset(MCTObject* self) {
  Py_BEGIN_ALLOW_THREADS
  self->tree.reset();
  Py_END_ALLOW_THREADS
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
  Py_BEGIN_ALLOW_THREADS
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
  Py_END_ALLOW_THREADS
  Py_XINCREF(Py_None);
  return Py_None;
}

static PyObject* gen_play(MCTObject* self, PyObject* args) {
  go_engine::Move move(go_engine::BLACK);
  Py_BEGIN_ALLOW_THREADS
  int debug_log = 0;
  if (!PyArg_ParseTuple(args, "p", &debug_log)) {
    return nullptr;
  }
  move = self->tree.gen_play(debug_log);
  Py_END_ALLOW_THREADS
  return PyLong_FromUnsignedLong(move.id());
}

static PyObject* score(MCTObject* self) {
  double s = self->tree.score();
  return PyFloat_FromDouble(s);
}
}  // namespace MCTPyBinding

static PyMethodDef MCT_methods[] = {
  {"reset", (PyCFunction)MCTPyBinding::reset, METH_NOARGS, "Reset the tree."},
  {"get_search_count", (PyCFunction)MCTPyBinding::get_search_count, METH_NOARGS, "Return the search / play out count of the current game state, this should always be called right after gen_play and before play."},
  {"is_valid", (PyCFunction)MCTPyBinding::is_valid, METH_VARARGS, "is_valid(color, pos): Test if a move is valid."},
  {"play", (PyCFunction)MCTPyBinding::play, METH_VARARGS, "play(color, pos): Play a move and change internal state."},
  {"gen_play", (PyCFunction)MCTPyBinding::gen_play, METH_VARARGS, "Gnerate a play using MCTS."},
  {"score", (PyCFunction)MCTPyBinding::score, METH_NOARGS, "Get my score - opponent's score."},
  {nullptr},
};

static PyTypeObject mct_py_type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "mcts.Tree",
  sizeof(MCTPyBinding::MCTObject),  // tp_basicsize
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

static PyObject* board_size(PyObject*, PyObject*) {
  return PyLong_FromLong((long)go_engine::N);
}

static PyMethodDef module_methods[] = {
  {"board_size", board_size, METH_NOARGS, "Get board size."},
  {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef mcts_module = {
  PyModuleDef_HEAD_INIT,
  "mcts",
  nullptr,
  -1,
  module_methods,
};

PyMODINIT_FUNC PyInit_mcts(void) {
  import_array();

  if (PyType_Ready(&eval_bridge_py_type) < 0) {
    return nullptr;
  }
  if (PyType_Ready(&mct_py_type) < 0) {
    return nullptr;
  }

  PyObject* m = PyModule_Create(&mcts_module);

  Py_INCREF(&eval_bridge_py_type);
  PyModule_AddObject(m, "EvalBridge", (PyObject*)&eval_bridge_py_type);
  Py_INCREF(&mct_py_type);
  PyModule_AddObject(m, "Tree", (PyObject*)&mct_py_type);
  return m;
}
