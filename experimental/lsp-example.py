import os

# import socket
# import json

hostname = "0.0.0.0"
port = 8080

file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
file_name = "test.py"
file_path = os.path.join(file_dir, file_name)


# def connect_to_server() -> any:
#     # Create a socket object
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     # Connect to the server
#     sock.connect((hostname, port))
#     request = {
#         "jsonrpc": "2.0",
#         "id": 1,
#         "method": "textDocument/didOpen",
#         "params": {
#             "textDocument": {
#                 "uri": f"file://{file_path}",
#                 "languageId": "python",
#                 "version": 1,
#                 "text": ""
#             }
#         }
#     }
#     encoded_request = json.dumps(request).encode("utf-8")
#     sock.sendall(encoded_request)

#     response = sock.recv(4096).decode("utf-8")
#     return response

# print(connect_to_server())
from monitors4codegen.multilspy import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import MultilspyConfig
from monitors4codegen.multilspy.multilspy_logger import MultilspyLogger

config = MultilspyConfig.from_dict({"code_language": "python"})
logger = MultilspyLogger()
lsp = SyncLanguageServer.create(config, logger, file_dir)

with lsp.start_server():
   result = lsp.request_definition(
       file_name, # Filename of location where request is being made
       22, # line number of symbol for which request is being made
       11 # column number of symbol for which request is being made
   )
   # ...
