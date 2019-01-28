// -*- mode:c++; c-basic-offset:2 -*-
#include <Python.h>
#include "board.h"

typedef struct {
  PyObject_HEAD
  go_engine::BoardInfo go_board;
} BoardObject;

static PyObject* Board_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  BoardObject* self = (BoardObject*)(type->tp_alloc(type, 0));
  return (PyObject*)self;
}

static int Board_init(BoardObject* self, PyObject* args, PyObject* kwargs);

static void Board_dealloc(BoardObject* self) {
  self->go_board.~BoardInfo();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Board_reset(BoardObject* self) {
  self->go_board.reset();
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* Board_debugString(BoardObject* self) {
  const std::string& s = self->go_board.DebugString();
  PyObject* ret = PyUnicode_DecodeUTF8(s.c_str(), s.size(), nullptr);
  return ret;
}

static PyObject* Board_score(BoardObject* self) {
  double s = self->go_board.score();
  PyObject* ret = PyFloat_FromDouble(s);
  return ret;
}

static PyObject* Board_is_valid(BoardObject* self, PyObject* args) {
  int color, pos;
  if (!PyArg_ParseTuple(args, "ii", &color, &pos)) {
    return nullptr;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "1st arg (color) can only be 0 or 1.");
    return nullptr;
  }
  if (pos < 0 || pos >= (int)go_engine::TotalMoves) {
    PyErr_SetString(PyExc_ValueError, "2nd arg (position) can only be [0, N * N).");
    return nullptr;
  }
  go_engine::Move move((go_engine::Color)color, pos);
  if (self->go_board.is_valid(move)) {
    Py_XINCREF(Py_True);
    return Py_True;
  } else {
    Py_XINCREF(Py_False);
    return Py_False;
  }
}

static PyObject* Board_play(BoardObject* self, PyObject* args) {
  int color, pos;
  if (!PyArg_ParseTuple(args, "ii", &color, &pos)) {
    return nullptr;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "1st arg (color) can only be 0 or 1.");
    return nullptr;
  }
  if (pos < 0 || pos >= (int)go_engine::TotalMoves) {
    PyErr_SetString(PyExc_ValueError, "2nd arg (position) can only be [0, N * N).");
    return nullptr;
  }
  go_engine::Move move((go_engine::Color)color, pos);
  self->go_board.play(move);
  Py_XINCREF(Py_None);
  return Py_None;
}

static PyObject* Board_has_stone(BoardObject* self, PyObject* args) {
  int color, pos;
  if (!PyArg_ParseTuple(args, "ii", &color, &pos)) {
    return nullptr;
  }
  if (color != go_engine::BLACK && color != go_engine::WHITE) {
    PyErr_SetString(PyExc_ValueError, "1st arg (color) can only be 0 or 1.");
    return nullptr;
  }
  if (pos < 0 || pos + 1 >= (int)go_engine::TotalMoves) {
    PyErr_SetString(PyExc_ValueError, "2nd arg (position) can only be [0, N * N).");
    return nullptr;
  }
  if (self->go_board.has_stone(pos, (go_engine::Color)color)) {
    Py_XINCREF(Py_True);
    return Py_True;
  } else {
    Py_XINCREF(Py_False);
    return Py_False;
  }
}

static PyMethodDef Board_methods[] = {
  {"reset", (PyCFunction)Board_reset, METH_NOARGS, "Reset the board."},
  {"debug", (PyCFunction)Board_debugString, METH_NOARGS, "Generate a debug string representing the board."},
  {"score", (PyCFunction)Board_score, METH_NOARGS, "Get black's score - white's score using Tromp-Taylor rules."},
  {"is_valid", (PyCFunction)Board_is_valid, METH_VARARGS, "is_valid(color, pos): Test if a move is valid."},
  {"play", (PyCFunction)Board_play, METH_VARARGS, "play(color, pos): Play a move."},
  {"has_stone", (PyCFunction)Board_has_stone, METH_VARARGS, "has_stone(color, pos): Test if a location has a stone of a specific color."},
  {nullptr},
};

static PyTypeObject board_BoardType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "board.Board",
  sizeof(BoardObject),  // tp_basicsize
  0,  // tp_itemsize
  (destructor)Board_dealloc,  // tp_dealloc
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
  "Go Board 9x9 Objects",  // tp_doc
  0,  // tp_traverse
  0,  // tp_clear
  0,  // tp_richcompare
  0,  // tp_weaklistoffset
  0,  // tp_iter
  0,  // tp_iternext

  Board_methods,  // tp_methods
  0,  // tp_members
  0,  // tp_getset
  0,  // tp_base
  0,  // tp_dict
  0,  // tp_descr_get
  0,  // tp_descr_set
  0,  // tp_dictoffset
  (initproc)Board_init,  // tp_init
  0,  // tp_alloc
  Board_new,  // tp_new
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

static int Board_init(BoardObject* self, PyObject* args, PyObject* kwargs) {
  const char* err_msg = "__init__() takes exactly one float as komi.";
  if (PyTuple_Size(args) != 1) {
    PyErr_SetString(PyExc_TypeError, err_msg);
    return 0;
  }
  PyObject* obj = PyTuple_GetItem(args, 0);
  if (!PyFloat_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, err_msg);
    return 0;
  }
  float komi = PyFloat_AsDouble(obj);
  if (komi <= 0 || int(komi) == komi) {
    PyErr_SetString(PyExc_TypeError, "Komi must be positive and not an exact integer.");
    return 0;
  }
  new(&(self->go_board)) go_engine::BoardInfo(komi);
  return 0;
}

static PyObject* board_system(PyObject* board, PyObject* args) {
  const char* command(nullptr);
  int sts;
  if (!PyArg_ParseTuple(args, "s", &command)) {
    return nullptr;
  }
  sts = system(command);
  return PyLong_FromLong(sts);
}

static PyMethodDef moduleMethods[] = {
  {"system", board_system, METH_VARARGS, "Execute a shell command."},
  {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef boardmodule = {
  PyModuleDef_HEAD_INIT,
  "board",
  nullptr,
  -1,
  moduleMethods,
};

PyMODINIT_FUNC PyInit_board(void) {
  // board_BoardType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&board_BoardType) < 0) {
    return nullptr;
  }
  PyObject* m = PyModule_Create(&boardmodule);
  Py_INCREF(&board_BoardType);
  PyModule_AddObject(m, "Board", (PyObject*)&board_BoardType);
  return m;
}
