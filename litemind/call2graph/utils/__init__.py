#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 17:57
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
from typing import Text
import os


def invalid(text: Text) -> bool:
    """
    判断text是否为无效文本
    :param text:
    :return:
    """
    if text is None or text.strip() == '':
        return True
    else:
        return False


# 文件上传处理逻辑
"""
@app.route("/litemind/demo1/parse_from_file", methods=["GET", "POST"])
@response_format
def parse_from_file():
    case_data = request.files.get('case_data')
    bot_id = request.files.get("bot_id", "11_1537515617")
    query_params = {"bot_id": bot_id}
    if not case_data:
        return "speech data is null", "REQUSET_PARAMS_MISS"

    file_name = request.form.get("file_name", "0")
    print(file_name)
    file_path = os.path.join('data/demo1/upload', file_name)
    case_data.save(file_path)
    with open(file_path, 'rb') as f:
        case_data = f.read()
        chart = chardet.detect(case_data)
        case_data = case_data.decode(encoding=chart['encoding'], errors="ignore")

    logging.debug(case_data)
    logging.debug(query_params)
    try:
        result = current_app.interpreter.parse(case_data, request_info=query_params)
    except Exception as e:
        logging.exception(e)
        return "内部解析错误", "QUERY_FAIL"
    logging.debug(result)
    return result, "DEFAULT_SUCCESS"
    
"""

# 文件上传
# <!DOCTYPE HTML>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# </head>
# <body>
#
# <form id="uploadform" method="post" enctype="multipart/form-data">
#     <label for="file">Select a file</label>
#     <input name="file" type="file">
#     <button id="submit" type="button">Upload</button>
# </form>
# <p>Result Filename:&nbsp;<span id="resultFilename"> here</span></p >
# <p>Result Filesize:&nbsp;<span id="resultFilesize">here</span></p >
#
# </body>
# </html>
