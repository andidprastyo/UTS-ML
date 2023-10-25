# UTS Machine Learning - Kelompok 1

**Anggota Kelompok :**

1. 2141720185 - Adam Rafi Rezandi
2. 2141720003 - Tio Misbaqul Irawan
3. 2141720019 - Bima Bayu Saputra
4. 2141720036 - Lailatul Badriyah
5. 2141720046 - Andi Dwi Prastyo

## Ketentuan UTS

Berdasarkan pemaparan kasus, Anda diminta untuk,

1. Pilih 5 citra plat nomor untuk setiap anggota kelompok dari dataset yang telah disediakan. DOWNLOAD
2. Lakukan segmentasi pada citra plat nomor untuk memperjelas karakter pada plat nomor.
3. Anda dapat menggunakan algortima K-Means seperti yang telah dijelaskan pada praktikum sebelumnya atau menggunakan algoritma klasterisasi yang lain.
4. Anda diperkenankan untuk melakukan pra pengolahan data (preprocessing) pada citra seperti,
   - Merubah color space
   - Reduksi dimensi
   - dsb
5. Tampilkan perbandingan citra antara sebelum dan sesudah di segmentasi

**Catatan:**

- Proses loading citra dicontohkan dengan menggunakan library openCV
- Secara default, openCV akan memuat citra dalam format BGR

## Hasil Segmentasi dari Berbagai Plat Nomer - Bima Bayu Saputra

**Citra Plat Original:**

![alt text](UTS_Bima/docs/Original.png)

Menggunakan 3 jenis preprocessing yang berbeda yaitu:

- Mengubah color space menjadi HSV
- Mereduksi citra menggunakan PCA
- Menggunakan Gaussian Blur dan Thresholding

### Mengubah color space menjadi HSV

```python
# Konversi ke ruang warna HSV
hsv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in original_images]
```

#### Hasil Preprossesing HSV

![alt text](UTS_Bima/docs/hsv.png)

### Mereduksi citra menggunakan PCA

```python
# Reduksi dimensi menggunakan PCA
pca = PCA(n_components=3)
reduced_images = [pca.fit_transform(img.reshape((-1, 3))).reshape(img.shape) for img in original_images]
```

#### Hasil Preprossesing PCA

![alt text](UTS_Bima/docs/pca.png)

### Menggunakan Gaussian Blur dan Thresholding

```python
# Mengubah citra ke grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in original_images]
# Kombinasi Thresholding dan Smoothing pada saluran kecerahan
preprocessed_images = [cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in gray_images]
```

#### Hasil Preprossesing Gaussian Blur dan Thresholding

![alt text](UTS_Bima/docs/ts.png)

### Melakukan Segmentasi dengan K-Means

```python
# K-Means untuk citra rgb
def kmeans_segmentation_color(image, k):
    # Ubah citra menjadi array 2D
    pixels = image.reshape(-1, 3).astype(np.float32)
    # Normalisasi nilai piksel
    pixels /= 255.0
    # Inisialisasi K-Means dengan k cluster
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=0)
    kmeans.fit(pixels)
    # Prediksi label setiap piksel
    labels = kmeans.predict(pixels)
    # Rekonstruksi citra hasil segmentasi
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)
    # Kembalikan citra hasil segmentasi
    segmented_image = (segmented_image * 255).astype(np.uint8)
    return segmented_image

# K-Means untuk citra grayscale
def kmeans_segmentation_gray(image, k):
    # Ubah citra grayscale menjadi array 2D
    pixels = image.reshape(-1, 1).astype(np.float32)
    # Inisialisasi K-Means dengan k cluster
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=0)
    kmeans.fit(pixels)
    # Prediksi label setiap piksel
    labels = kmeans.predict(pixels)
    # Rekonstruksi citra hasil segmentasi
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)
    # Kembalikan citra hasil segmentasi
    return segmented_image
```

```python
# Lakukan segmentasi pada citra-citra yang telah di-preprocess
segmented_images_hsv = [kmeans_segmentation_color(img, 2) for img in hsv_images]
# Lakukan segmentasi pada citra-citra yang telah di-preprocess
segmented_images_pca = [kmeans_segmentation_color(img, 2) for img in reduced_images]
# Lakukan segmentasi pada citra-citra yang telah di-preprocess
segmented_images_ts = [kmeans_segmentation_gray(img, 2) for img in preprocessed_images]
```

### Perbandingan Plat Asli dan Hasil Segementasi

![alt text](UTS_Bima/docs/hasil.png)

### Melakukan Evaluasi (Challenge )

```python
# Fungsi untuk menghitung akurasi pengenalan karakter
def calculate_accuracy(ground_truth, recognized_text):
    correct_characters = sum(1 for gt_char, rec_char in zip(ground_truth, recognized_text) if gt_char == rec_char)
    total_characters = len(ground_truth)
    accuracy = correct_characters / total_characters * 100.0
    return accuracy

# Fungsi untuk menghitung Levenshtein distance
def calculate_levenshtein_distance(ground_truth, recognized_text):
    return lev_distance(ground_truth, recognized_text)
```

#### Hasil Evaluasi Dengan Plat 'E 5234 YF'

![alt text](UTS_Bima/docs/eval2.png)

![alt text](UTS_Bima/docs/evalBM7098V.png)

#### Hasil Evaluasi Dengan Plat 'E 5234 YF'

![alt text](UTS_Bima/docs/eval.png)

![alt text](UTS_Bima/docs/evalE5234YF.png)

#### Hasil Evaluasi Dengan Plat 'E 3083 PAH'

![alt text](UTS_Bima/docs/eval1.png)

![alt text](UTS_Bima/docs/evalE3083PAH.png)

## Segmentasi dari Berbagai Plat Nomer - Lailatul Badriyah

### Menampilkan Citra Plat Nomor

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/original-image.png)

### Melihat Dimensi Citra Pertama

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/dimensi-citra.png)

### Normalisasi Data Citra Pertama

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/normalisasi-citra.png)

### Plot Distribusi Warna Piksel-Piksel dalam Citra

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/plot-pixels.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-plot.png)

### Segmentasi Citra Menggunakan Algoritma K-Means

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/segmentasi-kmeans.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-kmeans.png)

### Konversi Hasil Segmentasi K-Means Ke Biner Image

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/konversi-biner.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-biner.png)

Untuk empat gambar lainnya, prosesnya sama seperti di atas. Berikut saya sertakan hasil segmentasinya:

### Citra ke-2

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-2.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-biner-2.png)

### Citra ke-3

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-1.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-biner-1.png)

### Citra ke-4

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-3.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-biner-3.png)

### Citra ke-5

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-4.png)

![alt text](Lailatul%20Badriyah%20-%20UTS/docs/hasil-segmentasi-biner-4.png)

# Segmentasi Gambar Menggunakan K-Means Clustering - Andi

![alt text](./UTS%20-%20Andi/KMeans.png)

# Penggunaan Thresholding - Andi

![alt text](./UTS%20-%20Andi/Thresholding.png)

## Hasil Segmentasi dari Berbagai Plat Nomer - Adam Rafi Rezandi

### Citra Plat Original

Plat Nomor 1
![alt text](./UTS_Adam/data/img1.jpg)

Plat Nomor 2
![alt text](./UTS_Adam/data/img2.jpg)

Plat Nomor 3
![alt text](./UTS_Adam/data/img3.jpg)

Plat Nomor 4
![alt text](./UTS_Adam/data/img4.jpg)

Plat Nomor 5
![alt text](./UTS_Adam/data/img5.jpg)

### Citra Plat Setelah di Preprocessing

Plat Nomor 1
![alt text](./UTS_Adam/output/results/result1.png)

Plat Nomor 2
![alt text](./UTS_Adam/output/results/result2.png)

Plat Nomor 3
![alt text](./UTS_Adam/output/results/result3.png)

Plat Nomor 4
![alt text](./UTS_Adam/output/results/result4.png)

Plat Nomor 5
![alt text](./UTS_Adam/output/results/result5.png)

### Citra Plat dengan kotak bounding box

Plat Nomor 1
![alt text](./UTS_Adam/output/boundeds/bounded1.png)

Plat Nomor 2
![alt text](./UTS_Adam/output/boundeds/bounded2.png)

Plat Nomor 3
![alt text](./UTS_Adam/output/boundeds/bounded3.png)

Plat Nomor 4
![alt text](./UTS_Adam/output/boundeds/bounded4.png)

Plat Nomor 5
![alt text](./UTS_Adam/output/boundeds/bounded5.png)

### Hasil Evaluasi

```
Plate 1
Pixel Accuracy: 0.8391571428571428
Jaccard Index: 0.12823786081620722
Dice Coefficient: 0.22732415791016874

Plate 2
Pixel Accuracy: 0.8443214285714286
Jaccard Index: 0.12984158322379624
Dice Coefficient: 0.22984033363919398

Plate 3
Pixel Accuracy: 0.8319928571428571
Jaccard Index: 0.10979663216709379
Dice Coefficient: 0.1978680219144196

Plate 4
Pixel Accuracy: 0.7854071428571429
Jaccard Index: 0.1768293095145211
Dice Coefficient: 0.3005181942442761

Plate 5
Pixel Accuracy: 0.8196571428571429
Jaccard Index: 0.18755117749172973
Dice Coefficient: 0.31586205469959355
```

### Percobaan Recognition dengan Pytessearct

```Plate 1 : BE 2775 U

Ground 1 : BE 2775 U
Seberapa mirip: 1.0
Plate 2 : BE 7/179 U

Ground 2 : BE 7775 U
Seberapa mirip: 0.7368421052631579
Plate 3 : BE Q775 U

Ground 3 : BE 9775 U
Seberapa mirip: 0.8888888888888888
Plate 4 :
Ground 4 : BG 1980 A
Seberapa mirip: 0.0
Plate 5 : BM 3095 Vy

Ground 5 : BM 3095 V
Seberapa mirip: 0.9473684210526315
```
