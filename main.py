import asyncio
import datetime
import json
import logging
import numpy as np
import sqlite3
import sys
from socket import AF_INET, SOCK_STREAM, socket
from typing import Dict, Any

# MNE Signal Processing Integration
try:
    import mne
    mne.set_log_level('ERROR') # Suppress verbose MNE logging in standard output
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Skyfield Astrometric Integration
try:
    from skyfield.api import load, Topos
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

# Configure Application Logging
logging.basicConfig(
    filename="environmental_node.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CognitiveMNE Node")


class MNEBehavioralTelemetryNode:
    def __init__(self, bind_ip: str = "100.64.0.0", port: int = 8080, db_path: str = "telemetry.db"):
        self.bind_ip = bind_ip
        self.port = port
        self.db_path = db_path
        self.running = False
        
        self._init_database()
        
        if SKYFIELD_AVAILABLE:
            self.ts = load.timescale()
            self.planets = load('de421.bsp')
            self.earth = self.planets['earth']
        else:
            logger.warning("Skyfield library missing. Falling back to default baseline coordinate system.")

    def _init_database(self):
        """Initializes the SQLite relational persistence layer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mne_telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                hex_payload TEXT,
                human_text TEXT,
                alpha_power REAL,
                beta_power REAL,
                sky_elevation REAL,
                equilibrium_index REAL
            )
        ''')
        conn.commit()
        conn.close()

    def generate_mne_eeg_metrics(self) -> Dict[str, float]:
        """
        Uses MNE to construct a multi-channel EEG array, applies standard 
        signal filtering, and computes band power values using NumPy array operations.
        """
        if not MNE_AVAILABLE:
            # Fallback simulated spectrum if MNE is not installed
            return {"delta": 2.1, "theta": 5.4, "alpha": 10.2, "beta": 18.5, "gamma": 35.0, "equilibrium_ratio": 0.55}

        # Parameters: 4 channels (F3, F4, C3, C4), 1000 Hz sampling rate, 1 second duration
        sfreq = 1000  
        n_channels = 4
        ch_names = ['F3', 'F4', 'C3', 'C4']
        ch_types = ['eeg'] * n_channels
        
        # Create synthetic signal raw array (white noise + sine components)
        times = np.linspace(0, 1, sfreq)
        data = np.random.randn(n_channels, sfreq) * 1e-6 # 1 microvolt noise base
        
        # Inject Alpha (10 Hz) and Beta (20 Hz) signals across frontal channels
        data[0] += 5e-6 * np.sin(2 * np.pi * 10 * times) # F3 Alpha
        data[1] += 3e-6 * np.sin(2 * np.pi * 20 * times) # F4 Beta
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply bandpass filter across human EEG spectrum (0.5 Hz - 45 Hz)
        raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
        
        filtered_data = raw.get_data()
        
        # Extract mean band amplitudes
        alpha_power = float(np.mean(np.abs(filtered_data[:, 8:13]))) * 1e6
        beta_power = float(np.mean(np.abs(filtered_data[:, 13:30]))) * 1e6
        theta_power = float(np.mean(np.abs(filtered_data[:, 4:8]))) * 1e6
        
        # Compute equilibrium index: ratio of relaxing (Alpha) to high-stress (Beta) power
        equilibrium_ratio = alpha_power / (beta_power + 1e-6)
        
        return {
            "alpha": alpha_power,
            "beta": beta_power,
            "theta": theta_power,
            "equilibrium_ratio": equilibrium_ratio
        }

    def compute_sky_elevation(self) -> float:
        """Computes current celestial coordinate position via Skyfield."""
        if SKYFIELD_AVAILABLE:
            t = self.ts.now()
            observer = self.earth + Topos('49.2827 N', '123.1207 W')
            sun = self.planets['sun']
            astrometric = observer.at(t).observe(sun)
            alt, az, distance = astrometric.apparent().altaz()
            return float(alt.degrees)
        return 45.00

    def _persist_mccos(self, data: Dict[str, Any]):
        with open("node_state.mccos", "a") as f:
            f.write(json.dumps(data) + "\n")

    def _persist_txt(self, log_line: str):
        with open("environmental_audit.txt", "a") as f:
            f.write(log_line + "\n")

    async def telemetry_loop(self):
        self.running = True
        logger.info("Initiating MNE-integrated behavioral telemetry loop.")
        
        while self.running:
            timestamp = datetime.datetime.utcnow().isoformat()
            eeg_metrics = self.generate_mne_eeg_metrics()
            sky_elev = self.compute_sky_elevation()
            
            # Formulate behavioral status indicator based on MNE ratio
            status = "OPTIMAL_SAFETY" if eeg_metrics["equilibrium_ratio"] > 0.8 else "ADJUSTING_BASELINE"
            
            human_text = (
                f"TIME: {timestamp} | SKY ELEV: {sky_elev:.2f}° | "
                f"MNE ALPHA: {eeg_metrics['alpha']:.3f}µV | MNE BETA: {eeg_metrics['beta']:.3f}µV | "
                f"EQ_RATIO: {eeg_metrics['equilibrium_ratio']:.2f} | STATUS: {status}"
            )
            
            # Format outputs
            encoded_bytes = human_text.encode('utf-8')
            hex_payload = encoded_bytes.hex()
            binary_payload = ' '.join(format(b, '08b') for b in encoded_bytes[:28])
            
            # Terminal Output
            sys.stdout.write("\033[94m" + "="*80 + "\033[0m\n")
            sys.stdout.write(f"\033[1m[HUMAN READABLE]:\033[0m {human_text}\n")
            sys.stdout.write(f"\033[36m[HEXADECIMAL   ]:\033[0m {hex_payload[:64]}...\n")
            sys.stdout.write(f"\033[33m[BINARY STREAM ]:\033[0m {binary_payload}...\n")
            sys.stdout.flush()
            
            # File persistence (.txt, .mccos, .db)
            self._persist_txt(human_text)
            self._persist_mccos({
                "timestamp": timestamp,
                "eeg_mne": eeg_metrics,
                "sky_elevation": sky_elev,
                "hex": hex_payload
            })
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mne_telemetry_logs (timestamp, hex_payload, human_text, alpha_power, beta_power, sky_elevation, equilibrium_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, hex_payload, human_text, eeg_metrics['alpha'], eeg_metrics['beta'], sky_elev, eeg_metrics['equilibrium_ratio'])
            )
            conn.commit()
            conn.close()
            
            await asyncio.sleep(2.0)

    async def start_listener(self):
        try:
            server = socket(AF_INET, SOCK_STREAM)
            server.bind((self.bind_ip, self.port))
            server.listen(5)
            server.setblocking(False)
            logger.info(f"Socket successfully bound to interface {self.bind_ip}:{self.port}")
            print(f"\n[NODE ACTIVE] Listener initialized at {self.bind_ip}:{self.port}")
        except Exception as e:
            logger.error(f"Socket binding deferred on {self.bind_ip}:{self.port} -> {e}")
            print(f"\n[NOTICE] Network interface {self.bind_ip} not bound directly. Running asynchronous telemetry pipeline.")

    async def run(self):
        await self.start_listener()
        await self.telemetry_loop()

if __name__ == "__main__":
    node = MNEBehavioralTelemetryNode(bind_ip="100.64.0.0", port=8080)
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] MNE Telemetry node stopped successfully.")
