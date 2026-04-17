# FashCortex Inswapper - Proje Analizi ve Notlar

## 1. Sistemin Genel Amacı ve Yapısı
Bu proje, `runpod-python` kütüphanesini kullanarak hazırlanmış sunucusuz (serverless) bir RunPod "worker" servisidir. Fashplay veya benzeri uygulamalarda gönderilen "kaynak yüzü (source)", "hedef modele (target)" giydirmek (face-swap) için kullanılır.

Kullanılan Temel Teknolojiler ve Modeller:
- **InsightFace & Inswapper:** Yüz tespiti ve yüz değiştirme için (`inswapper_128.onnx` ve `buffalo_l` modelleri).
- **CodeFormer:** Yüz değiştirme sonrası oluşan bozulmaları gidermek ve yüz çözünürlüğünü artırmak için.
- **Real-ESRGAN:** Arka planı ve genel resmi büyütmek (upscale) için.

## 2. Nasıl Çalışır? (İşlem Akışı)
1. Cihaz başlatıldığında (Docker ayağa kalkarken veya RunPod aktifleştiğinde) `__main__` metodu çalışır. FaceSwap, FaceAnalyser ve CodeFormer/Real-ESRGAN modelleri bir kere belleğe/VRAM'e (`TORCH_DEVICE = 'cuda'`) yüklenir.
2. Dışarıdan RunPod `/run` veya `/runsync` servisine base64 formatında JSON içeren bir API isteği gelir.
3. `handler.py` içindeki `face_swap_api` metodu bu isteği alır, base64 kodlarını decode edip `/tmp/` dizinine `.png`/`.jpg` olarak yazar.
4. `process()` fonksiyonu; hem hedef hem de kaynak görsellerdeki yüzlerin konumlarını `buffalo_l` ile tespit eder.
5. Algılanan yüzlere karşılıklı olarak `swap_face()` fonksiyonu çağrılarak atama yapılır (inswapper).
6. İşlem bittikten sonra sonuç fotoğrafı üretilir. Eğer API isteğinde `face_restore=True` olarak geldiyse, resim `restoration.py`'e gönderilir ve `CodeFormer` ile yüz yüksek kalite formata yükseltilir.
7. Çıkan nihai sonuç, tekrar base64 string'e dönüştürülerek kullanıcıya JSON içerisinde response (yanıt) olarak iletilir ve `/tmp/` dizinindeki dosyalar silinerek temizlik yapılır.

## 3. Minimum GPU Gereksinimi
- Yalnızca `inswapper` (yüz değiştirme) işlemi fazla GPU belleği (VRAM) gerektirmez. Ancak işlem bitiminde tetiklenen **CodeFormer** ve **Real-ESRGAN** (Eğer `face_restore` ve upscale aktifse) ani VRAM tüketimine sebep olur.
- Özellikle çözünürlüğü yüksek (1000px üstü) görsellerde ve yoğun işlem anında OutOfMemory (OOM - bellek taşması) almamak için **Minimum 8GB - 12GB VRAM**'e sahip kartlar gereklidir (örneğin RTX 3060/3070/4060).
- RunPod'da "production" ortamı için en dengeli fiyat/performans ürünü **RTX A4000 (16GB)** olacaktır.

## 4. Hızlandırma ve Gelişim Noktaları (Optimizasyonlar)
Bu proje genel olarak temiz yazılmış ancak önemli hızlandırma ve optimizasyon fırsatları barındırıyor:

### a) Gereksiz Disk I/O (Girdi/Çıktı) İşleminin Kaldırılması 
Şu anki akışta API'den gelen resim (base64) önce fiziki diske yazılıyor, ardından tekrar okunuyor:
```python
# Mevcut Durum: base64 -> /tmp/file.png -> Image.open(path)
```
Diske yazıp okuma işlemi yerine, her şey doğrudan RAM'de (in-memory) tutularak milisaniyelik gecikmelerden ve disk kullanımından tasarruf edilebilir.
```python
# Önerilen:
import io
source_img = Image.open(io.BytesIO(base64.b64decode(source_image_data)))
```
Böylece kod içerisindeki `clean_up_temporary_files` süreçlerine, try-catch içi temizleme karmaşasına ve disk okuma-yazma hızına olan bağımlılığa gerek kalmayacaktır.

### b) TensorRT Kullanımı (Daha Hızlı Inswapper)
`handler.py`'de InsightFace modelleri başlatılırken sadece `CUDAExecutionProvider` kullanılıyor.
Inswapper (`.onnx` modeli), NVIDIA kartlarda TensorRT ile %30 ile %50 arası daha hızlı çalışabilir. Dockerfile üzerinden TensorRT paketlerini kurup `providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']` olarak güncellenmesi düşünülebilir. (Ancak cold-start sırasında ilk derleme motoru [engine/plan] oluşturulacağından ilk istek yavaş dönebilir, kalıcı [persistent] depolama gerekir)

### c) Yüz Algılama Min Size Filtresi (Min Face Size)
`get_many_faces` içindeki `min_face_size` filtresi faydalı olsa da hedef imaj için (target image) devre dışı bırakılmış. Sistem, uzaktaki alakasız bulanık yüzleri de algılayıp değiştirmeye çalışarak zaman kaybedebilir.

### d) Batch (Çoklu) İşleme Eksikliği
`process` fonksiyonu içerisinde hedefler ve kaynaklar arası geçişler döngü ile tek tek yapılıyor (`for i in range(num_iterations):`). InsightFace modellemesi buna el veriyorsa (ya da koda eklenecekse), birden fazla yüz aynı anda batch (yığın) işleme sokulursa hız artar.

### e) CodeFormer Alternatifleri
Eğer hız (FPS / Response Time) en önemli unsursa, restorasyon aşamasında ağır çalışan CodeFormer ve Real-ESRGAN ikilisi yerine, daha hızlı çalışan hafif **GFPGAN** veya **RestoreFormer** gibi modellere geçiş yapılabilir.

## Sonuç
Bu worker şimdiki haliyle production'da işini görecek stabiliteye sahip. Geçici dosya yazma-okuma sistemini bellek üzerinden `io.BytesIO` kullanarak güncellemek ilk aşamada alacağınız en büyük ücretsiz hız kazanımı olacaktır.
