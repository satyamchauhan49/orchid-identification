import io
import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from PIL import Image
from image_app.server import decode,handler

class UploadChecks(unittest.TestCase):
    def image(self):
        b=io.BytesIO();Image.new('RGB',(20,30)).save(b,format='PNG');return b.getvalue()
    def test_png_decode(self):self.assertEqual(decode(self.image()).size,(20,30))
    def test_bad_file(self):
        with self.assertRaises(ValueError):decode(b'not an image')
    def test_gif_rejected(self):
        b=io.BytesIO();Image.new('RGB',(20,20)).save(b,format='GIF')
        with self.assertRaises(ValueError):decode(b.getvalue())
    def test_server_origin_and_actual_image_request(self):
        class Model:
            demo=True
            def infer(self,data):return {'size':decode(data).size,'demo':True}
        server=HTTPServer(('127.0.0.1',0),handler(Model()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            host='127.0.0.1:'+str(server.server_port)
            c=HTTPConnection(host)
            c.request('GET','/');r=c.getresponse();self.assertEqual(r.status,200);r.read()
            c.request('POST','/api/predict',self.image(),{'Content-Type':'image/png','Origin':'https://untrusted.example'})
            r=c.getresponse();self.assertEqual(r.status,403);r.read()
            c.request('POST','/api/predict',self.image(),{'Content-Type':'image/png','Origin':'http://'+host})
            r=c.getresponse();self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['size'],[20,30])
            c.close()
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
