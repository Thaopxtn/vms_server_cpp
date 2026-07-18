import socket
import uuid
import json
import sys

def scan():
    msg = f"""<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns="http://www.w3.org/2003/05/soap-envelope" xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
  <Header>
    <wsa:MessageID>urn:uuid:{uuid.uuid4()}</wsa:MessageID>
    <wsa:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>
    <wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>
  </Header>
  <Body>
    <Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <Types xmlns:dn="http://www.onvif.org/ver10/network/wsdl">dn:NetworkVideoTransmitter</Types>
    </Probe>
  </Body>
</Envelope>"""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(3.0)

    try:
        sock.sendto(msg.encode('utf-8'), ('239.255.255.250', 3702))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

    found = set()
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            found.add(addr[0])
        except socket.timeout:
            break
        except Exception:
            break

    print(json.dumps(list(found)))

if __name__ == '__main__':
    scan()
