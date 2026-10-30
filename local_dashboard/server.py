"""Local archive UI with bounded public capture; never executes models or orders."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit,parse_qs,unquote
from http.cookies import SimpleCookie
from pathlib import Path
import argparse,ipaddress,json,mimetypes,secrets,socket,subprocess,functools
import data
import collector

HERE=Path(__file__).resolve().parent
DIST=HERE/'dist'
@functools.lru_cache(maxsize=1)
def lan_addresses():
    result=set()
    # Hostname resolution can point to a VPN's benchmarking adapter on Windows.
    # Read actual interface addresses, without modifying network or firewall state.
    if __import__('os').name=='nt':
        try:
            p=subprocess.run(['powershell.exe','-NoProfile','-Command',
                "Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred | Select-Object -ExpandProperty IPAddress"],
                capture_output=True,text=True,timeout=6,creationflags=0x08000000)
            for value in p.stdout.splitlines():
                value=value.strip()
                if value.startswith(('10.','192.168.')) or value.startswith('172.') and 16<=int(value.split('.')[1])<=31:result.add(value)
        except (OSError,ValueError,subprocess.TimeoutExpired):pass
    for a in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET):
        ip=a[4][0]
        if ipaddress.ip_address(ip).is_private and not ip.startswith(('127.','169.254.','198.18.','198.19.')): result.add(ip)
    return sorted(result)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send(self,body,content_type='application/json; charset=utf-8',status=200,cookie=None):
        if isinstance(body,dict):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status)
        for k,v in {'Content-Type':content_type,'Content-Length':str(len(body)),
                    'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
                    'Referrer-Policy':'no-referrer','X-Frame-Options':'DENY'}.items():self.send_header(k,v)
        if cookie:self.send_header('Set-Cookie',cookie)
        self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        parts=urlsplit(self.path);qs=parse_qs(parts.query);ip=ipaddress.ip_address(self.client_address[0])
        if not (ip.is_private or ip.is_loopback):return self.send({'error':'仅限本机与局域网'},status=403)
        cookie=SimpleCookie()
        try:cookie.load(self.headers.get('Cookie',''))
        except Exception:pass
        provided=qs.get('key',[''])[0] or (cookie['dashboard_key'].value if 'dashboard_key' in cookie else '')
        valid=secrets.compare_digest(provided,self.server.access_key)
        if not ip.is_loopback and not valid:return self.send({'error':'请使用启动器提供的手机访问链接'},status=403)
        auth_cookie=('dashboard_key='+self.server.access_key+'; HttpOnly; SameSite=Strict; Path=/') if valid else None
        path=unquote(parts.path)
        try:
            if path=='/api/health':return self.send(dict(app='龙头战法本地看板',mode='ARCHIVE_WITH_AUTO_PUBLIC_CAPTURE',catalog=data.catalog(),lan=lan_addresses(),collector_enabled=True),cookie=auth_cookie)
            if path=='/api/live-status':return self.send(self.server.collector.status(),cookie=auth_cookie)
            if path=='/api/catalog':return self.send(data.catalog(),cookie=auth_cookie)
            if path=='/api/report':
                day=qs.get('date',[data.catalog()['latest']])[0]
                return self.send(data.report(day),cookie=auth_cookie)
            if path.startswith('/evidence/'):
                seg=path.split('/')
                if len(seg)!=4 or seg[3]!='relationship.png' or seg[2] not in data.archives():raise KeyError(path)
                image=data.archives()[seg[2]]/'minutes/情绪关系分时图.png'
                return self.send(image.read_bytes(),'image/png',cookie=auth_cookie)
            asset=(DIST/('index.html' if path=='/' else path.lstrip('/'))).resolve()
            if not asset.is_relative_to(DIST.resolve()) or not asset.is_file():raise KeyError(path)
            return self.send(asset.read_bytes(),mimetypes.guess_type(str(asset))[0] or 'application/octet-stream',cookie=auth_cookie)
        except (KeyError,FileNotFoundError):return self.send({'error':'未找到对应归档或文件'},status=404)
        except Exception as e:return self.send({'error':'读取失败；旧数据未替代本次结果','detail':str(e)[:180]},status=500)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);p.add_argument('--lan',action='store_true');args=p.parse_args()
    runtime=HERE/'.runtime';runtime.mkdir(exist_ok=True)
    keyfile=runtime/'access_key.txt'
    if not keyfile.exists():keyfile.write_text(secrets.token_urlsafe(24),encoding='utf-8')
    key=keyfile.read_text(encoding='utf-8').strip()
    host='0.0.0.0' if args.lan else '127.0.0.1'
    server=ThreadingHTTPServer((host,args.port),Handler);server.access_key=key
    server.collector=collector.Collector();server.collector.start()
    print('Local: http://127.0.0.1:'+str(args.port)+'/',flush=True)
    if args.lan:
        for ip in lan_addresses():print('LAN: http://'+ip+':'+str(args.port)+'/?key='+key,flush=True)
    print('Archive review + automatic free public-data status; no model execution. Ctrl+C stops.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.collector.stop();server.server_close()

if __name__=='__main__':main()
