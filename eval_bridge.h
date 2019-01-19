// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#ifndef INCLUDE_GUARD_EVAL_BRIDGE_H__
#define INCLUDE_GUARD_EVAL_BRIDGE_H__

#include <array>
#include <atomic>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <semaphore.h>

#include "board.h"
#include "debug_msg.h"

// This class accumulates pending eval requests from multiple threads, batch them and feed to the
// underlying eval engine (e.g., tensorflow) for better performance.
namespace mcts {
template<size_t LogBatchSize>
class NetworkEvalBridge {
  static constexpr size_t BatchSize = 1ULL << LogBatchSize;
  static constexpr size_t BoardSize = go_engine::N * go_engine::N;
  static constexpr size_t BatchCopies = 16;
  static constexpr size_t BufferSize = BatchCopies * BatchSize;
public:
  NetworkEvalBridge(PyObject* _eval)
    : eval(_eval)
  {
    CHECK(PyCallable_Check(eval)) << "Python object is not callable: " << PyUnicode_AsASCIIString(PyObject_Str(eval));
    Py_XINCREF(eval);
    npy_intp dims[4] = {BatchSize, 3, go_engine::N, go_engine::N};
    for (size_t i = 0; i < BatchCopies; ++i) {
      args[i] = PyTuple_New(1);
      PyObject* array_obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT, input_buffer.data() + get_slot_offset(i * BatchSize));
      PyTuple_SetItem(args[i], 0, array_obj);
    }
    sem_init(&eval_start, 0, 0);
    for (size_t i = 0; i < BatchCopies; ++i) {
      sem_init(&eval_done[i], 0, 0);
      sem_init(&batch_done[i], 0, BatchSize);
    }
  }

  ~NetworkEvalBridge() {
    Py_XDECREF(eval);
    for (size_t i = 0; i < BatchCopies; ++i) {
      Py_XDECREF(args[i]);
    }
    sem_destroy(&eval_start);
    for (size_t i = 0; i < BatchCopies; ++i) {
      sem_destroy(&eval_done[i]);
      sem_destroy(&batch_done[i]);
    }
  }
  // Implementing copy constructor requires proper deep copy and handling of reference counting of Python objects.
  template<typename... Dummy> NetworkEvalBridge(Dummy...) = delete;

  // # of threads calling operator() (eval) must be >= BatchSize and < 2 * BatchSize.
  size_t worker_thread_count() {
    return BatchSize * 1.5;
  }

  void startEval(PyThreadState *_save) {
    // Event loop.
    while (true) {
      sem_wait(&eval_start);
      std::atomic_thread_fence(std::memory_order_acquire);

      uint64_t id = batch_id;
      batch_id = -1;
      CHECK(id < BatchCopies) << "Invalid batch_id: " << id;
      PyEval_RestoreThread(_save);
      PyObject* result = PyObject_CallObject(eval, args[id]);
      if (result == nullptr) {
        PyErr_PrintEx(1);
        CHECK(false) << "Failed calling Python eval function: nullptr returned.";
      }
      ASSERT(PyTuple_Check(result));
      ASSERT(PyTuple_Size(result) == 2) << "Callback returns a tuple of size " << PyTuple_Size(result);

      PyObject* policy_result = PyTuple_GetItem(result, 0);
      PyObject* value_result = PyTuple_GetItem(result, 1);
      Py_XINCREF(policy_result);
      Py_XINCREF(value_result);
      Py_XDECREF(result);

      ASSERT(PyArray_Check(policy_result)) << "Return value 1 is not PyArray.";
      ASSERT(PyArray_Check(value_result)) << "Return value 2 is not PyArray.";
      if (policy_output[id]) {
        Py_XDECREF(policy_output[id]);
      }
      if (value_output[id]) {
        Py_XDECREF(value_output[id]);
      }
      _save = PyEval_SaveThread();
      policy_output[id] = (PyArrayObject*)policy_result;
      value_output[id] = (PyArrayObject*)value_result;
      // Check output type & shape.
      {
        ASSERT(PyArray_TYPE(policy_output[id]) == NPY_FLOAT) << "Elements in returned PyArray are type " << PyArray_TYPE(policy_output[id]) << ", expecting " << NPY_FLOAT;
        ASSERT(PyArray_NDIM(policy_output[id]) == 2) << "Returned PyArray has a dimension other than 2: " << PyArray_NDIM(policy_output[id]);
        npy_intp* dims = PyArray_DIMS(policy_output[id]);
        ASSERT(dims[0] == BatchSize && dims[1] == go_engine::TotalMoves)
          << "Returned PyArray has size: (" << dims[0] << ", " << dims[1] << "), expecting ("
          << BatchSize << ", " << go_engine::TotalMoves << ").";
      }
      {
        ASSERT(PyArray_TYPE(value_output[id]) == NPY_FLOAT) << "Elements in returned PyArray are type " << PyArray_TYPE(value_output[id]) << ", expecting " << NPY_FLOAT;
        ASSERT(PyArray_NDIM(value_output[id]) == 2) << "Returned PyArray has a dimension other than 2: " << PyArray_NDIM(value_output[id]);
        npy_intp* dims = PyArray_DIMS(value_output[id]);
        ASSERT(dims[0] == BatchSize && dims[1] == 1)
          << "Returned PyArray has size: (" << dims[0] << ", " << dims[1] << "), expecting (" << BatchSize << ", 1).";
      }

      std::atomic_thread_fence(std::memory_order_release);
      // Wake up all worker threads waiting for this batch.
      for (size_t i = 0; i < BatchSize; ++i) {
        sem_post(&eval_done[id]);
      }
    }  // while
  }

  // MCTS Worker threads call this function to queue eval requests.  The function blocks until
  // enough eval requests are accumulated so it can send them to the eval thread as a batch.
  //
  // Each slot has 3 states:
  //
  // 1. Unused.
  // 2. Input filled, waiting for eval.
  // 3. Eval done, waiting for output to be consumed.
  //
  // State change is a cycle: 1 -> 2 -> 3 -> 1.
  float operator()(const go_engine::BoardInfo& b, go_engine::Color color, std::array<float, go_engine::TotalMoves>& prior) {
    const uint64_t my_eval_id = eval_count.fetch_add(1, std::memory_order_relaxed);
    const uint64_t my_slot_id = my_eval_id % BufferSize;
    const uint64_t my_batch_id = my_slot_id / BatchSize;
    const uint64_t my_offset = get_slot_offset(my_slot_id);
    
    sem_wait(&batch_done[my_batch_id]);
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t m = 0; m < BoardSize; ++m) {
      input_buffer[my_offset + m] = b.has_stone(m, color);
      input_buffer[my_offset + m + BoardSize] = b.has_stone(m, go_engine::opposite_color(color));
      input_buffer[my_offset + m + 2 * BoardSize] = color;
    }
    if (input_filled[my_batch_id].fetch_add(1, std::memory_order_release) + 1 == BatchSize) {
      // I'm the last one finishing this batch, so notify the eval thread.
      batch_id = my_batch_id;
      input_filled[my_batch_id].store(0, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_release);
      sem_post(&eval_start);
    }
    // Now wait for the eval thread.
    sem_wait(&eval_done[my_batch_id]);
    std::atomic_thread_fence(std::memory_order_acquire);

    // Copy eval result.
    memcpy(prior.data(), PyArray_GETPTR2(policy_output[my_batch_id], my_slot_id % BatchSize, 0), sizeof(float) * go_engine::TotalMoves);
    float ret = *(const float*)PyArray_GETPTR2(value_output[my_batch_id], my_slot_id % BatchSize, 0);

    if (input_filled[my_batch_id].fetch_add(1, std::memory_order_release) + 1 == BatchSize) {
      input_filled[my_batch_id].store(0, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_release);
      // Allow the next batch of threads to enter.
      for (size_t i = 0; i < BatchSize; ++i) {
        sem_post(&batch_done[my_batch_id]);
      }
    }
    return ret;
  }
private:
  size_t get_slot_offset(size_t slot) const {
    ASSERT(slot < BufferSize) << "Invalid slot: " << slot << " >= " << BufferSize;
    return slot * 3 * go_engine::N * go_engine::N;
  }

  PyObject* eval = nullptr;
  std::array<PyObject*, BatchCopies> args{};
  std::array<float, BufferSize * 3 * go_engine::N * go_engine::N> input_buffer{};
  std::array<PyArrayObject*, BatchCopies> policy_output{};
  std::array<PyArrayObject*, BatchCopies> value_output{};

  std::atomic<uint64_t> eval_count = 0;
  std::array<std::atomic<uint64_t>, BatchCopies> input_filled{};

  // These are used to signal the eval thread when a batch of eval requests are fully filled.
  uint64_t batch_id = -1;
  // Only one eval is possible at a time.  This is not too much of a restriction since GPU likes
  // large batches.
  sem_t eval_start;

  // eval_done: Used by the eval thread to signal worker threads to fetch their respective eval
  // results and resume the work.
  std::array<sem_t, BatchCopies> eval_done;
  // batch_done: Signals once all workers finished copying the eval result out of the batch, so that
  // waiting workers can start fill new input data (strictly speaking it can be made more fine
  // grained since there is no conflict between filling input with new data and copying output).
  std::array<sem_t, BatchCopies> batch_done;
};
}  // namespace mcts

#endif // INCLUDE_GUARD_EVAL_BRIDGE_H__
