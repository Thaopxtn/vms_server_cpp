import sys
import os
import time
import subprocess
import json
import hashlib
import re
import urllib.parse
import requests
import xml.etree.ElementTree as ET
from requests.auth import HTTPBasicAuth, HTTPDigestAuth


def query_db(sql):
    env = os.environ.copy()
    env["PGPASSWORD"] = "585858"
    cmd = ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "smart_monitoring", "-t", "-A", "-c", sql]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")
        if res.returncode != 0:
            print(f"[POLLER ERROR] psql error: {res.stderr.strip()}", flush=True)
            return None
        return res.stdout.strip()
    except Exception as e:
        print(f"[POLLER ERROR] failed to run psql: {e}", flush=True)
        return None


def dahua_md5_hash(random_val, realm, username, password):
    str1 = f"{username}:{realm}:{password}"
    hash1 = hashlib.md5(str1.encode()).hexdigest().upper()
    str2 = f"{username}:{random_val}:{hash1}"
    hash2 = hashlib.md5(str2.encode()).hexdigest().upper()
    return hash2


def clean_host(host):
    if not host:
        return ""
    if host.startswith("http://"):
        host = host[7:]
    elif host.startswith("https://"):
        host = host[8:]
    if "/" in host:
        host = host.split("/")[0]
    return host.strip()


def is_port_reachable(host, port, timeout=3):
    import socket
    if not host or not port:
        return False
    clean_h = clean_host(host)
    if not clean_h:
        return False
    try:
        sock = socket.create_connection((clean_h, int(port)), timeout=timeout)
        sock.close()
        return True
    except Exception:
        return False


def parse_host_port_from_url(url):
    if not url:
        return None, None
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[-1]
        if ":" in netloc:
            h, p = netloc.split(":", 1)
            return h, int(p)
        default_p = 80 if (url.startswith("http://") or url.startswith("https://")) else 554
        return netloc, default_p
    except Exception:
        return None, None


def normalize_connection_state(value):
    raw = str(value or "").strip()
    lowered = raw.lower()
    if lowered in ("connected", "connect", "online", "true", "1", "active", "ok", "normal"):
        return "online", raw
    if lowered in ("disconnected", "disconnect", "offline", "false", "0", "inactive", "exception", "failed", "error"):
        return "offline", raw
    if "connected" in lowered or "online" in lowered:
        return "online", raw
    if "disconnected" in lowered or "offline" in lowered or "fail" in lowered or "error" in lowered:
        return "offline", raw
    return "offline", raw


def channel_to_zero_index(value):
    try:
        channel = int(str(value).strip())
    except Exception:
        return None
    if channel <= 0:
        return None
    if channel >= 100:
        channel = channel // 100
    return channel - 1


def parse_channel_from_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "channel" in query:
            return int(query["channel"][0])
    except Exception:
        pass

    m = re.search(r"[?&]channel=(\d+)", url)
    if m:
        return int(m.group(1))

    m = re.search(r"/Streaming/Channels/(\d+)", url, re.IGNORECASE)
    if m:
        channel_code = int(m.group(1))
        if channel_code >= 100:
            return channel_code // 100
        return channel_code

    return None


def request_with_recorder_auth(url, username, password, timeout=5):
    auth_methods = [
        HTTPDigestAuth(username or "", password or ""),
        HTTPBasicAuth(username or "", password or ""),
        None,
    ]
    last_error = None
    for auth in auth_methods:
        try:
            resp = requests.get(url, auth=auth, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)
    if last_error:
        print(f"[POLLER WARNING] Recorder request failed for {url}: {last_error}", flush=True)
    return None


def get_dahua_camera_states(host, port, username, password):
    clean_h = clean_host(host)
    login_url = f"http://{clean_h}:{port}/RPC2_Login"
    rpc_url = f"http://{clean_h}:{port}/RPC2"

    try:
        payload = {
            "method": "global.login",
            "params": {
                "userName": username,
                "password": "",
                "clientType": "Dahua3.0-Web3.0",
            },
            "id": 1,
        }
        resp = requests.post(login_url, json=payload, timeout=5)
        if resp.status_code != 200:
            return None, None

        data = resp.json()
        params = data.get("params", {})
        random_val = params.get("random")
        realm = params.get("realm")
        session_id = data.get("session")
        if not random_val or not realm or not session_id:
            return None, None

        hashed_pwd = dahua_md5_hash(random_val, realm, username, password)
        payload_login = {
            "method": "global.login",
            "params": {
                "userName": username,
                "password": hashed_pwd,
                "clientType": "Dahua3.0-Web3.0",
                "authorityType": "Default",
                "passwordType": "Default",
            },
            "session": session_id,
            "id": 2,
        }
        resp_login = requests.post(login_url, json=payload_login, timeout=5)
        if resp_login.status_code != 200:
            return None, None

        data_login = resp_login.json()
        session_id = data_login.get("session") or data_login.get("result")
        if not session_id or isinstance(session_id, bool):
            session_id = data_login.get("session") or payload_login["session"]

        payload_state = {
            "method": "LogicDeviceManager.getCameraState",
            "params": {"uniqueChannels": [-1]},
            "session": session_id,
            "id": 3,
        }
        resp_state = requests.post(rpc_url, json=payload_state, timeout=5)
        if resp_state.status_code != 200:
            return None, None

        data_state = resp_state.json()
        states_list = data_state.get("params", {}).get("states", [])
        states_map = {}
        for item in states_list:
            ch = item.get("channel")
            conn_state = item.get("connectionState")
            if ch is not None and conn_state:
                states_map[int(ch)] = conn_state
        if states_map:
            return states_map, "dahua_rpc"
    except Exception as e:
        print(f"[POLLER WARNING] Dahua NVR {clean_h}:{port} query failed: {e}", flush=True)
    return None, None


def xml_name(tag):
    return tag.split("}", 1)[-1].lower()


def parse_hikvision_status_xml(xml_text):
    states_map = {}
    root = ET.fromstring(xml_text)
    for node in root.iter():
        children = list(node)
        if not children:
            continue
        values = {}
        for child in children:
            text = (child.text or "").strip()
            if text:
                values[xml_name(child.tag)] = text

        channel_id = (
            values.get("id")
            or values.get("channelid")
            or values.get("inputid")
            or values.get("proxychannelid")
        )
        state = (
            values.get("online")
            or values.get("connectstatus")
            or values.get("connectionstatus")
            or values.get("status")
            or values.get("streamingstatus")
        )
        idx = channel_to_zero_index(channel_id)
        if idx is None or state is None:
            continue
        states_map[idx] = normalize_connection_state(state)[1]
    return states_map


def get_hikvision_camera_states(host, port, username, password):
    clean_h = clean_host(host)
    candidates = [
        f"http://{clean_h}:{port}/ISAPI/ContentMgmt/InputProxy/channels/status",
        f"http://{clean_h}:{port}/ISAPI/ContentMgmt/InputProxy/channels",
        f"http://{clean_h}:{port}/ISAPI/System/Video/inputs/channels",
        f"http://{clean_h}:{port}/ISAPI/Streaming/channels",
    ]
    for url in candidates:
        resp = request_with_recorder_auth(url, username, password)
        if not resp or not (resp.text or "").strip():
            continue
        try:
            states = parse_hikvision_status_xml(resp.text)
            if states:
                return states, f"hikvision_isapi:{url.rsplit('/', 1)[-1]}"
        except Exception as e:
            print(f"[POLLER WARNING] Hikvision XML parse failed for {url}: {e}", flush=True)
    return None, None


def get_direct_nvr_camera_states(dvr):
    brand = str(dvr.get("brand") or "").strip().lower()
    host = dvr.get("host")
    port = dvr.get("http_port")
    username = dvr.get("username")
    password = dvr.get("password")

    attempts = []
    if brand in ("dahua", "generic", ""):
        attempts.append(get_dahua_camera_states)
    if brand in ("hikvision", "hik", "hk", "generic", ""):
        attempts.append(get_hikvision_camera_states)
    if not attempts:
        attempts = [get_dahua_camera_states, get_hikvision_camera_states]

    for fn in attempts:
        states, source = fn(host, port, username, password)
        if states:
            return states, source

    # Fallback: Check if RTSP/TCP port or HTTP port of the device is reachable via TCP
    rtsp_port = dvr.get("rtsp_port") or 554
    http_port = dvr.get("http_port") or 80
    if is_port_reachable(host, rtsp_port) or is_port_reachable(host, http_port):
        print(f"[POLLER] NVR {host} ports ({rtsp_port}/{http_port}) are reachable via TCP. Using tcp_reachability fallback.", flush=True)
        fallback_map = {i: "Connected" for i in range(64)}
        return fallback_map, "tcp_reachability"

    return None, None


def build_status(status, source, dvr_id, channel=None, raw_state=None):
    item = {
        "status": status,
        "source": source,
        "dvr_id": dvr_id or "",
        "updated_at": int(time.time()),
    }
    if channel is not None:
        item["channel"] = channel
    if raw_state:
        item["raw_state"] = raw_state
    return item


def load_previous_statuses(target_path):
    try:
        if not os.path.exists(target_path):
            return {}
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("[POLLER] NVR Camera Status Poller started.", flush=True)

    os.makedirs("web", exist_ok=True)
    target_path = "web/camera_status.json"

    while True:
        try:
            dvr_json_str = query_db("SELECT json_agg(d) FROM (SELECT dvr_id, host, http_port, rtsp_port, username, password, brand FROM dvrs) d;")
            if not dvr_json_str or dvr_json_str == "[]" or dvr_json_str == "None":
                print("[POLLER] No DVRs found in database.", flush=True)
                time.sleep(15)
                continue

            dvrs = json.loads(dvr_json_str)

            cam_json_str = query_db("SELECT json_agg(c) FROM (SELECT camera_id, dvr_id, rtsp_url FROM cameras WHERE is_active=true) c;")
            if not cam_json_str or cam_json_str == "[]" or cam_json_str == "None":
                print("[POLLER] No active cameras found in database.", flush=True)
                time.sleep(15)
                continue

            cameras = json.loads(cam_json_str)
            previous_statuses = load_previous_statuses(target_path)

            nvr_states = {}
            nvr_sources = {}
            for dvr in dvrs:
                dvr_id = dvr.get("dvr_id")
                host = dvr.get("host")
                port = dvr.get("http_port")

                print(f"[POLLER] Querying NVR {dvr_id} ({host}:{port}) directly...", flush=True)
                states, source = get_direct_nvr_camera_states(dvr)
                if states:
                    nvr_states[dvr_id] = states
                    nvr_sources[dvr_id] = source or "nvr_direct"
                    print(f"[POLLER] NVR {dvr_id} returned {len(states)} channel states from {nvr_sources[dvr_id]}.", flush=True)
                else:
                    print(f"[POLLER WARNING] NVR {dvr_id} direct query returned no states.", flush=True)

            camera_statuses = {}
            for cam in cameras:
                cam_id = cam.get("camera_id")
                dvr_id = cam.get("dvr_id")
                rtsp_url = cam.get("rtsp_url") or ""

                if not dvr_id or dvr_id not in nvr_states:
                    cam_host, cam_port = parse_host_port_from_url(rtsp_url)
                    if cam_host and is_port_reachable(cam_host, cam_port):
                        camera_statuses[cam_id] = build_status("online", "direct_tcp_reachability", dvr_id)
                    else:
                        previous = previous_statuses.get(cam_id)
                        if isinstance(previous, dict) and previous.get("status"):
                            camera_statuses[cam_id] = {
                                **previous,
                                "source": "stale_" + str(previous.get("source") or "nvr"),
                                "stale": True,
                                "updated_at": int(time.time()),
                            }
                        else:
                            camera_statuses[cam_id] = build_status("offline", "nvr_unavailable", dvr_id)
                    continue

                channel_num = parse_channel_from_url(rtsp_url)
                if channel_num is None:
                    camera_statuses[cam_id] = build_status("offline", "channel_unknown", dvr_id)
                    continue

                nvr_ch_idx = channel_num - 1
                states_map = nvr_states[dvr_id]
                if nvr_ch_idx in states_map:
                    conn_state = states_map[nvr_ch_idx]
                    status, raw_state = normalize_connection_state(conn_state)
                    camera_statuses[cam_id] = build_status(
                        status,
                        nvr_sources.get(dvr_id, "nvr_direct"),
                        dvr_id,
                        channel_num,
                        raw_state,
                    )
                else:
                    camera_statuses[cam_id] = build_status("offline", "channel_missing_from_nvr", dvr_id, channel_num)

            temp_path = target_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(camera_statuses, f, indent=2, ensure_ascii=False)

            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)

            online_count = sum(1 for item in camera_statuses.values() if isinstance(item, dict) and item.get("status") == "online")
            print(f"[POLLER] Updated {len(camera_statuses)} camera statuses from NVR-first sources. Online: {online_count}.", flush=True)

        except Exception as e:
            print(f"[POLLER ERROR] Error in poller loop: {e}", flush=True)

        time.sleep(15)


if __name__ == "__main__":
    main()