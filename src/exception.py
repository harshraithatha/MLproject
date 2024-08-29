import sys

def error_message_detail(error, error_detail:sys):

    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno

    error_message = f'Error occured in python script name [{file_name}], line number [{line_num}], error message [{error}]'

    return error_message