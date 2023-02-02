

def clean_alt_list(list_):
    list_ = list_.strip('[ ]')
    list_ = list_.replace("'", '')
    list_ = list_.split(', ')

    return list_
