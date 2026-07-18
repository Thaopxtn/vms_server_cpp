// Native WebRTC implementation to replace go2rtc iframe
// Features: Silent failure (no UI logs), Auto-reconnect, Backoff, UI State Sync

class VMSWebRTCPlayer {
    constructor(videoElement, streamId, go2rtcUrl) {
        this.video = videoElement;
        this.streamId = streamId;
        this.go2rtcUrl = go2rtcUrl || `http://${location.hostname}:1984`;
        this.pc = null;
        this.reconnectTimer = null;
        this.reconnectAttempts = 0;
        this.isPlaying = false;
        
        // Mute by default, set object-fit
        this.video.muted = true;
        this.video.autoplay = true;
        this.video.playsInline = true;
        this.video.style.width = "100%";
        this.video.style.height = "100%";
        this.video.style.objectFit = "contain"; // Smart grid fit
        
        this.init();
    }

    notifyState(state) {
        // state: "loading", "online", "offline"
        const slot = this.video.closest(".cam-slot");
        if (!slot) return;
        
        const dot = slot.querySelector(".cam-status-dot");
        const overlay = slot.querySelector(".cam-offline-overlay");
        
        if (state === "online") {
            slot.classList.remove("offline");
            if (dot) { dot.classList.add("online"); dot.classList.remove("offline"); }
            if (overlay) overlay.style.display = "none";
        } else if (state === "offline") {
            slot.classList.add("offline");
            if (dot) { dot.classList.remove("online"); dot.classList.add("offline"); }
            if (overlay) {
                overlay.style.display = "flex";
                const txt = overlay.querySelector(".off-text");
                if (txt) txt.textContent = "OFFLINE";
            }
        } else if (state === "loading") {
            slot.classList.add("offline");
            if (dot) { dot.classList.remove("online"); dot.classList.add("offline"); }
            if (overlay) {
                overlay.style.display = "flex";
                const txt = overlay.querySelector(".off-text");
                if (txt) txt.textContent = "CONNECTING...";
            }
        }
    }

    async init() {
        if (this.pc) this.destroy();
        this.notifyState("loading");
        
        try {
            this.pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            
            this.pc.addTransceiver('video', { direction: 'recvonly' });
            this.pc.addTransceiver('audio', { direction: 'recvonly' });

            this.pc.ontrack = (event) => {
                if (this.video.srcObject !== event.streams[0]) {
                    this.video.srcObject = event.streams[0];
                    this.video.play().catch(e => console.warn("Autoplay blocked:", e));
                }
            };

            this.pc.onconnectionstatechange = () => {
                const state = this.pc.connectionState;
                if (state === 'failed' || state === 'disconnected' || state === 'closed') {
                    this.scheduleReconnect();
                } else if (state === 'connected') {
                    this.reconnectAttempts = 0; // reset
                    this.isPlaying = true;
                    this.notifyState("online");
                }
            };

            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            const url = `${this.go2rtcUrl}/api/webrtc?src=${encodeURIComponent(this.streamId)}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: offer.type,
                    sdp: offer.sdp
                })
            });

            if (!response.ok) throw new Error(`HTTP error ${response.status}`);
            
            const answer = await response.json();
            await this.pc.setRemoteDescription(new RTCSessionDescription(answer));

        } catch (error) {
            console.error(`[WebRTC] Connection failed for ${this.streamId}:`, error);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.isPlaying = false;
        this.notifyState("offline");
        
        // Exponential backoff: max 10 seconds
        this.reconnectAttempts++;
        const backoff = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 10000);
        
        this.reconnectTimer = setTimeout(() => {
            console.log(`[WebRTC] Reconnecting ${this.streamId}... (Attempt ${this.reconnectAttempts})`);
            this.init();
        }, backoff);
    }

    setMute(isMuted) {
        this.video.muted = isMuted;
    }

    changeStream(newStreamId) {
        this.streamId = newStreamId;
        this.init();
    }

    destroy() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        this.isPlaying = false;
        this.notifyState("offline");
    }
}

window.VMSWebRTCPlayer = VMSWebRTCPlayer;
