import http.server
import socketserver
import os

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/en', '/fr', '/nl']:
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

PORT = 5000

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()