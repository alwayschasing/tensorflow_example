#!/usr/bin/env python
# coding=utf-8

class CodeUtil(object):
    @staticmethod
    def UTF2GBK(str):
        string = str.decode('utf-8')
        unistr = CodeUtil.B2Q(string)
        rstring = unistr.encode('gb18030')
        return rstring

    @staticmethod
    def GBK2UTF(str):
        string = str.decode('gb18030')
        unistr = CodeUtil.Q2B(string)
        rstring = unistr.encode('utf-8')
        return rstring

    @staticmethod
    def B2Q(ustring):
        # ustring: unicode string
        rstring = ""
        for uchar in ustring:
            inside_code=ord(uchar)
            if inside_code == 32:
                inside_code = 12288
            elif inside_code > 32 and inside_code <= 126:
                inside_code += 65248
            rstring += unichr(inside_code)
        return rstring

    @staticmethod
    def Q2B(ustring):
        # ustring: unicode string
        rstring = ""
        for uchar in ustring:
            inside_code=ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += unichr(inside_code)
        return rstring

    # all normalization res is quanjiao, lower, gb18030
    @staticmethod
    def UTFNormalize(str):
        lw = str.lower() 
        tmp_str = lw.decode("utf-8")
        tmp_str = CodeUtil.B2Q(tmp_str)
        tmp_str =  tmp_str.encode("gb18030")
        return tmp_str
    @staticmethod
    def GBKNormalize(str):
        lw = str.lower()
        tmp_str = lw.decode("gb18030")
        tmp_str = CodeUtil.B2Q(tmp_str)
        tmp_str = tmp_str.encode("gb18030")
        return tmp_str

if __name__ == "__main__":
    pass

