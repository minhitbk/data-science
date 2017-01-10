""" 
 Written by Minh Tran <minhitbk@gmail.com>, Jan 2017.
 This file modifies codes to adapt to the software.
 Original source code reference: Tensorflow github.
"""
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.rnn import _rnn_step
from tensorflow.python.util import nest

from ..config.cf_container import Config

# pylint: disable=protected-access
_state_size_with_prefix = rnn_cell._state_size_with_prefix
# pylint: enable=protected-access


def _infer_state_dtype(explicit_dtype, state):
    """
    Infer the dtype of an RNN state.
    """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to "
                "infer a single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    """
    Creates a dynamic version of bidirectional recurrent neural network.
    The initial state for both directions is zero by default.
    :param cell_fw: An instance of RNNCell, to be used for forward direction
    :param cell_bw: An instance of RNNCell, to be used for backward direction
    :param inputs: The RNN inputs
    :param sequence_length: An int32/int64 vector
    :param initial_state_fw: An initial state for the forward RNN
    :param initial_state_bw: An initial state for the backward RNN
    :param dtype: The data type for the initial states and expected output
    :param parallel_iterations: The number of iterations in parallel
    :param swap_memory:
    :param time_major:
    :param scope:
    :return: A tuple (outputs, output_states)
    """
    if not isinstance(cell_fw, rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory, time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = array_ops.reverse_sequence(
                input=inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp_output_bw, tmp_output_state_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse,
                sequence_length=sequence_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory, time_major=time_major,
                scope=bw_scope)

    output_bw = array_ops.reverse_sequence(
        input=tmp_output_bw, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    if Config.cell_type == "LSTMCell":
        tmp_output_state_bw_ = tmp_output_state_bw.c
    else:
        tmp_output_state_bw_ = tmp_output_state_bw

    output_state_bw = array_ops.reverse_sequence(
        input=tmp_output_state_bw_, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)

    if Config.cell_type == "LSTMCell":
        output_states = (output_state_fw.c, output_state_bw)
    else:
        output_states = (output_state_fw, output_state_bw)

    return (outputs, output_states)


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    """
    Creates a recurrent neural network specified by RNNCell `cell`.
    :param cell: An instance of RNNCell
    :param inputs: The RNN inputs
    :param sequence_length: An int32/int64 vector sized
    :param initial_state: An initial state for the RNN
    :param dtype: The data type for the initial state and expected output
    :param parallel_iterations: The number of iterations in parallel
    :param swap_memory:
    :param time_major:
    :param scope:
    :return: A tuple (outputs, output_states)
    """
    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
        # (B,T,D) => (T,B,D)
        flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                           for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.get_shape())
        sequence_length = array_ops.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
        batch_size = input_shape[0][1]

        for input_ in input_shape:
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError("All inputs should have the same batch size")

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    "If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.pack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs,
                                       flat_sequence=flat_input)

        (outputs, states) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            flat_output = nest.flatten(outputs)
            flat_output = [array_ops.transpose(output, [1, 0, 2])
                           for output in flat_output]
            flat_state = nest.flatten(states)
            flat_state = [array_ops.transpose(state_, [1, 0, 2])
                          for state_ in flat_state]

            outputs = nest.pack_sequence_as(
                structure=outputs, flat_sequence=flat_output)
            states = nest.pack_sequence_as(
                structure=states, flat_sequence=flat_state)

        return (outputs, states)


def _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations,
                      swap_memory, sequence_length=None, dtype=None):
    """
    Internal implementation of Dynamic RNN.
    :param cell: An instance of RNNCell
    :param inputs: A Tensor of shape [time, batch_size, input_size]
    :param initial_state: A Tensor of shape [batch_size, state_size]
    :param parallel_iterations: Positive Python int
    :param swap_memory:
    :param sequence_length:
    :param dtype:
    :return: A tuple (outputs, output_states)
    """
    state = initial_state
    assert isinstance(parallel_iterations,
                      int), "parallel_iterations must be int"

    state_size = cell.state_size
    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)
    flat_state_size = nest.flatten(cell.state_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape "
                "inference, but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input "
                "in a batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _state_size_with_prefix(size, prefix=[batch_size])
        return array_ops.zeros(
            array_ops.pack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i, _infer_state_dtype(
        dtype, state)) for i in range(len(flat_output_size)))

    state_ta = tuple(_create_ta("state_%d" % i, _infer_state_dtype(
        dtype, state)) for i in range(len(flat_state_size)))

    input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unpack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state_ta_t, state):
        """
        Take a time step of the dynamic RNN.
        """
        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        # Pack state if using state tuples
        output = nest.flatten(output)
        new_state_ = nest.flatten(new_state)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))
        state_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(state_ta_t, new_state_))

        return (time + 1, output_ta_t, state_ta_t, new_state)

    _, output_final_ta, state_final_ta, fin_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state_ta, state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.pack() for ta in output_final_ta)
    final_states = tuple(ta.pack() for ta in state_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _state_size_with_prefix(
            output_size, prefix=[const_time_steps, const_batch_size])
        output.set_shape(shape)

    for state, state_size in zip(final_states, flat_state_size):
        shape = _state_size_with_prefix(
            state_size, prefix=[const_time_steps, const_batch_size])
        state.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)
    final_states = nest.pack_sequence_as(
        structure=cell.state_size, flat_sequence=final_states)

    return (final_outputs, final_states)
