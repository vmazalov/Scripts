import cntk as C
import numpy as np

'''
Implementation of GRID LSTM in CNTK
'''
C.device.try_set_default_device(C.device.gpu(0))

def freq_grid(input, output_dim, slice_size=10, slice_overlap=5):
    # slice the input vector along frequency
    input_dim = input.shape[0]

    right_ind = slice_size
    # array with freq outputs at the prev time step
    m_t_1_k_list = []
    c_t_1_k_list = []

    while (right_ind <= input_dim):
        name1 = 'm_t_1_k' + str(right_ind)
        m_t_1_k_list.append(C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes, name=name1))
        name1 = 'c_t_1_k' + str(right_ind)
        c_t_1_k_list.append(C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes, name=name1))
        right_ind = right_ind + slice_overlap

    left_ind = 0
    right_ind = slice_size
    k_ind = 0
    GLSTM_cell_list = []
    GLSTM_cell = grid_lstm_factory(slice_size, output_dim)
    while (right_ind <= input_dim):
        freq_slice = C.slice(input, 0, left_ind, right_ind)
        if k_ind == 0:
            f_x_h_c = GLSTM_cell(m_t_1_k_list[k_ind], C.Constant(0, (output_dim)), c_t_1_k_list[0], C.Constant(0, (output_dim)), freq_slice)
        else:
            f_x_h_c = GLSTM_cell(m_t_1_k_list[k_ind], GLSTM_cell_list[k_ind-1].outputs[1], c_t_1_k_list[k_ind], GLSTM_cell_list[k_ind-1].outputs[3], freq_slice)

        GLSTM_cell_list.append(f_x_h_c)

        right_ind = right_ind + slice_overlap
        left_ind = left_ind + slice_overlap
        k_ind = k_ind + 1

    result = C.splice(C.combine([GLSTM_cell_list[0].outputs[0]]), C.combine([GLSTM_cell_list[0].outputs[1]]))
    i = 0
    while i < k_ind:
        replacements = {m_t_1_k_list[i] : C.sequence.past_value(GLSTM_cell_list[i].outputs[0]).output, c_t_1_k_list[i] : C.sequence.past_value(GLSTM_cell_list[i].outputs[2]).output}
        GLSTM_cell_list[i].replace_placeholders(replacements)
        result = C.splice(result, C.combine([GLSTM_cell_list[i].outputs[0]]), C.combine([GLSTM_cell_list[i].outputs[1]]))
        i = i + 1

    assert((right_ind - slice_overlap) == input_dim)

    return result

def GLSTM_layer(input, output_dim):
    input_dim=input.shape[0]
    # we first create placeholders for the hidden state and cell state which we don't have yet
    m_t_1_k = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    m_tk_1  = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    c_t_1_k = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    c_tk_1  = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)

    # we now create an LSTM_cell function and call it with the input and placeholders
    GLSTM_cell = grid_lstm_factory(input_dim, output_dim) # C.layers.LSTM(output_dim) #
    f_x_h_c = GLSTM_cell(m_t_1_k, m_tk_1, c_t_1_k, c_tk_1, input)
    h_c = f_x_h_c.outputs

    replacements = { m_t_1_k: C.sequence.past_value(h_c[0]).output, m_tk_1: C.sequence.past_value(h_c[1]).output, c_t_1_k: C.sequence.past_value(h_c[2]).output, c_tk_1: C.sequence.past_value(h_c[3]).output  }
    f_x_h_c.replace_placeholders(replacements)

    # and finally we return the hidden state and cell state as functions (by using `combine`)
    return C.combine([f_x_h_c.outputs[0]]), C.combine([f_x_h_c.outputs[1]]), C.combine([f_x_h_c.outputs[2]]), C.combine([f_x_h_c.outputs[3]])

def grid_lstm_factory(input_dim, output_dim, init=C.glorot_uniform(), init_bias = 0):
    # (11)
    W_t_ix  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_t_ix')
    W_k_ix  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_k_ix')

    W_t_im  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_im')
    W_k_im  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_im')

    W_t_ic  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_ic')
    W_k_ic  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_ic')

    b_t_i   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_t_i')
    b_k_i   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_k_i')

    # (12)
    W_t_fx  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_t_fx')
    W_k_fx  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_k_fx')

    W_t_fm  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_fm')
    W_k_fm  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_fm')

    W_t_fc  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_fc')
    W_k_fc  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_fc')

    b_t_f   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_t_f')
    b_k_f   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_k_f')

    # (13), (14)
    W_t_cx  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_t_cx')
    W_k_cx  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_k_cx')

    W_t_cm  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_cm')
    W_k_cm  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_cm')

    b_t_c   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_t_c')
    b_k_c   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_k_c')

    # (15)
    W_t_ox  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_t_ox')
    W_k_ox  = C.parameter(shape=(input_dim, output_dim), init=init, name='W_k_ox')

    W_t_om  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_om')
    W_k_om  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_om')

    W_t_oc  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_t_oc')
    W_k_oc  = C.parameter(shape=(output_dim,output_dim), init=init, name='W_k_oc')

    b_t_o   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_t_o')
    b_k_o   = C.parameter(shape=(output_dim), init = init_bias, name = 'b_k_o')

    def grid_lstm_func(m_t_1_k, m_tk_1, c_t_1_k, c_tk_1, x_tk):
        common_11 = C.times(m_t_1_k, W_t_im) + C.times(m_tk_1, W_k_im) + C.times(c_t_1_k, W_t_ic) + C.times(c_tk_1, W_k_ic)
        i_t_tk = C.sigmoid(C.times(x_tk, W_t_ix) + common_11 + b_t_i)
        i_k_tk = C.sigmoid(C.times(x_tk, W_k_ix) + common_11 + b_k_i)

        common_12 = C.times(m_t_1_k, W_t_fm) + C.times(m_tk_1, W_k_fm) + C.times(c_t_1_k, W_t_fc) + C.times(c_tk_1, W_k_fc)
        f_t_tk = C.sigmoid(C.times(x_tk, W_t_fx) + common_12 + b_t_f)
        f_k_tk = C.sigmoid(C.times(x_tk, W_k_fx) + common_12 + b_k_f)

        c_t_tk = C.element_times(f_t_tk,c_t_1_k) + C.element_times(i_t_tk,C.tanh(C.times(x_tk, W_t_cx) + C.times(m_t_1_k, W_t_cm) + C.times(m_tk_1, W_k_cm) + b_t_c))   # (13)
        c_k_tk = C.element_times(f_k_tk,c_tk_1) + C.element_times(i_k_tk,C.tanh(C.times(x_tk, W_k_cx) + C.times(m_t_1_k, W_t_cm) + C.times(m_tk_1, W_k_cm) + b_k_c))   # (14)

        common_15 = C.times(m_t_1_k, W_t_om) + C.times(m_tk_1, W_k_om) + C.times(c_t_tk, W_t_oc) + C.times(c_k_tk, W_k_oc)
        o_t_tk = C.sigmoid(C.times(x_tk, W_t_ox) + common_15 + b_t_o)
        o_k_tk = C.sigmoid(C.times(x_tk, W_k_ox) + common_15 + b_k_o)

        m_t_tk = C.element_times(o_t_tk, C.tanh(c_t_tk))
        m_k_tk = C.element_times(o_k_tk, C.tanh(c_k_tk))

        return (m_t_tk, m_k_tk, c_t_tk, c_k_tk)

    return C.BlockFunction("grid_lstm", "tf_grid")(grid_lstm_func)