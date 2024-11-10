# Web-App-Real-Time-Vehicle-and-Statistics

### *Abstract*
Proyek ini adalah proyek Internship Data Scientist di PT Global Data Inspirasi, Proyek ini adalah aplikasi web untuk deteksi kendaraan secara real-time dan visualisasi statistik lalu lintas. Proyek ini menggunakan model YOLOv10 untuk melakukan deteksi dan pelacakan berbagai jenis kendaraan, seperti mobil, bus, truk, dan sepeda motor. Fitur utama dari proyek ini meliputi penghitungan objek secara real-time, estimasi kecepatan kendaraan, dan visualisasi data secara langsung menggunakan Plotly.js. Sistem ini terintegrasi dengan Flask untuk menampilkan hasil deteksi kendaraan secara real-time serta memproses data menggunakan threading dan HLS untuk penyimpanan video secara efisien.

  <a href="#">
    <img alt="Web-App-Real-Time-Vehicle-and-Statistics" src="https://github.com/BagasDwiSantosa/Web-App-Real-Time-Vehicle-and-Statistics/blob/master/Screenshot_12-10-2024_183922_127.0.0.1.jpeg" />
  </a>

### Berikut adalah langkah-langkah singkat untuk menjalankan proyek **Web App Real-Time Vehicle Detection and Statistics**:
1. **Buat Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate   # Linux/MacOS
   env\Scripts\activate      # Windows
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Buat Struktur Folder**
   ```bash
   mkdir templates img static model
   ```

4. **Jalankan Aplikasi Flask**
   ```bash
   python Web-App-Real-Time-Vehicle-and-Statistics.py
   ```

