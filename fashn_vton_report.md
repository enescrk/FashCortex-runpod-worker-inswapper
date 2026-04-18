# FASHN VTON 1.5 - Kapsamlı Proje İnceleme Raporu

**Proje Adı:** FASHN VTON v1.5
**Mevcut Sürüm:** 1.5.0
**Lisans:** Apache-2.0
**Kategori:** Maskesiz Sanal Giyim (Maskless Virtual Try-On), Difüzyon Modelleri, Üretken Yapay Zeka (GenAI)

---

## 1. Genel Bakış ve Temel Özellikler
FASHN VTON v1.5, **maske gerektirmeyen (maskless)** ve doğrudan "pixel space" (piksel uzayında) çalışan, son derece verimli bir sanal kıyafet deneme modelidir. Geleneksel VTON (Virtual Try-On) sistemlerinin aksine kullanıcının çok dikkatli bir maskeleme veya segmentasyon yapmasına gerek kalmadan, yapay zeka ile otomatik vücut ayrıştırması yaparak çalışır.

*   **Maskesiz (veya Esnek Maskeli) Üretim:** Modelin en dikkat çeken özelliği `--segmentation-free` desteği sunmasıdır. Bu özellik sayesinde mevcut kıyafetin hacmine (volume) bağlı kalmadan, yeni giyilecek kıyafet bol veya farklı kesimdeyse bile vücutta doğal durması sağlanır.
*   **İki Farklı Kıyafet Fotoğrafı Tipi:** Model, kıyafet (garment) görüntüsünü iki modda yorumlayabiliyor: 
    *   `model`: Başka bir insanın (mankenin) üzerinde duran kıyafet.
    *   `flat-lay`: Düz bir zemine serilmiş, stüdyo ürün fotoğrafı. Düz ürün fotoğraflarında dummy (sahte) poz anahtar noktaları kullanılarak üretim sağlanır.
*   **Çoklu Kategori Modu:** Model *tops* (üst giyim), *bottoms* (alt giyim) ve *one-pieces* (tek parça, elbise vs.) kategorilerini destekleyecek şekilde `garment_categories` (kategori etiketleri) alabilir.

## 2. Model Mimarisi ve Teknoloji Yığını

### 2.1 Makine Öğrenmesi Çerçevesi (Framework)
Projenin temel iskeleti **PyTorch** üzerine kurulmuştur. Özellikle yeni nesil **MMDiT (Multi-Modal Diffusion Transformer)** mimarisini kullanmaktadır. Bu altyapı meşhur `FLUX.1` modelinden esinlenmiş ve modifiye edilmiştir.

### 2.2 Çekirdek Bileşenler (Core Components)
Projenin uçtan uca (end-to-end) akışı (`pipeline.py`) aşağıdaki 3 ana modeli kullanır:
1.  **TryOnModel (MMDiT - Multi-Modal Diffusion Transformer):** Orijinal FLUX benzeri mimaridir. Hem kişi görüntüsünü (`DoubleStreamBlock`), hem de giydirilecek kıyafetin görüntünü (text encoder yerine ikinci stream olarak) birleştirerek Difüzyon üretimini yapar.
2.  **DWPose (Pose Detector):** İnsan duruşunu (iskelet, el ve yüz hatlarını) hesaplar. Yüksek veri işleme başarımı için `onnxruntime-gpu` (YOLOX ve DW-LL Onnx modelleri) ile çalışmaktadır.
3.  **FashnHumanParser:** İnsan silüetini ve giydiği mevcut kıyafeti ayrıştırmak (segmentation) için kullanılır.

### 2.3 Ön İşleme (Preprocessing) Aşaması
`fashn_vton/__call__` akışı şöyledir:
*   Kullanıcı resmi ve kıyafet resmi yüklenir. Boyutları yapısal bozulmayı engellemek için `AspectPreserveResize` ile en yüksek kenar baz alınarak (örneğin 864px VEYA ayarlanan maksimum boyut) yeniden boyutlandırılır.
*   **DWPose** ile insan fotoğrafından duruş (pose) noktaları çıkartılır. Eğer kıyafet de bir modele giydirilmişse aynı işlem ondan da çıkarılır.
*   **Human Parser** fotoğrafı inceler; kafa, kollar, bacaklar, üst giyim vb. ayrıştırılır. Hangi kategorinin (*örn: tops*) deneneceğine bağlı olarak orijinal kıyafet resimden silinerek `ca_image` (Clothing Agnostic Image - Kıyafetsiz Bırakılmış Görüntü) oluşturulur.
*   **Tensor Hazırlığı:** Bütün çıktılar Tensöre çevrilir ve -1 ile 1 arasında normalize edilir (uint8 -> neg1_1).

### 2.4 Try-On MMDiT Ağı (Ağ Yapısı)
*   `tryon_mmdit.py` içerisinde yer alan model; `EmbedND` (Rotary Positional Embeddings - RoPE), QK Normları ve Ölçeklendirilmiş Çarpım Dikkat Modülü (PyTorch native `scaled_dot_product_attention`) kullanmaktadır.
*   Veri seti işlenirken Transformer mimarisinde tipik olan *PatchEmbed* kullanılarak resimler bloklar/patch dizileri haline getirilir (`patch_size=12` kullanılmış).
*   **DoubleStreamBlocks:** Kişi resmi ile (ca_image + maskeler) kıyafet resmi bağımsız streamler halinde işlenir ancak bu blok içindeki capraz dikkat (cross-attention) ile birbiriyle haberleşerek harmanlanır. Bu kısım doku (texture) aktarımını çok yüksek kalitede tutar.

### 2.5 Difüzyon Sampler ve Zaman Çizelgesi
İsteğe bağlı Euler sampling döngüsünde adım adım (default 30 adım) difüzyon denoise işlemi çalışır. Özellikle son adımlarda (N adım) renk doygunluk patlamasını (color saturation) engellemek için **CFG (Classifier-Free Guidance)** es geçilmekte veya zayıflatılmaktadır (`skip_cfg_last_n_steps=1`).

## 3. Donanım, Başarım ve Optimize Edilebilirlik
*   **bfloat16 (BF16) Desteği:** Model, mimarisi gereği `bfloat16` destekleyen RTX 30xx, 40xx, A100 ve H100 kartlarında doğrudan bellek ve performans açısından yarı boyutla çalışır. Kart `bf16` desteklemiyorsa otomatik olarak `float32` rejimine geçer. `float16` yerine `bfloat16` kullanmaları gradyan limit patlamaları yerine difüzyon üretim doğruluğunu artırır.
*   **VRAM İhtiyacı:** Safetensors boyutu ve Onnx modelleriyle beraber ortalama *10-12 GB civarında* bir VRAM ihtiyacı olduğu tahmin edilmektedir.
*   **Kullanılabilir Boyut (Resolution):** `input_shape=(864, 576)` olarak belirlenmiş varsayılan bir baz eğitim çözünürlüğü bulunur (yatay 576, dikey 864 form faktörü).

## 4. Dosya Ağacı ve Önemli Skriptler
*   `scripts/download_weights.py:` Tüm ana modelleri (safetensors model ve DWPose ONNX ağları) otomatik olarak indirir (~2GB dolaylarında). Human parser modelini ise HuggingFace hub üzerinden otomatik çeker.
*   `examples/basic_inference.py:` Sistem parametreleriyle argparser üzerinden CMD konsol komutlarını denemek için kullanışlıdır (`--num-samples`, `--guidance-scale`, `--category`, vb. kullanımdadır).
*   `pyproject.toml` ve `setup:` `pip install -e .` aracılığıyla projeyi sisteme hızlıca entegre eder.

## 5. Projenin Ekstreme Entegrasyonu (Sizin Altyapınıza Uygulanabilirlik)
Eğer FashCortex altyapınıza (ve Inswapper Runpod Workera) bunu entegre etmeniz gerekirse;
1.  **Boyut Standardizasyonu:** Bu model `fashn_vton` objesi olarak belleğe alındıktan sonra GPU RAM'inde hatırı sayılır yer kaplar. Inswapper işlemi ve VTON'u aynı pod içinde çalıştıracaksanız bellek yönetimine (örn: 24GB VRAM barındıran RTX 3090 / L4 / A5000 makinelerde çalışmasına) dikkat edin.
2.  **Codeformer ile VTON:** VTON sonrası insan yüzünün gerçekçiliğini yitirmemesi için Fashn-VTON sonrası çıkan resim Inswapper ve CodeFormer'dan geçirilirse, tam teşekküllü (Kıyafet + Kesin Hedef Yüz) mükemmel bir sonuç elde edilir.

## Özet
FASHN VTON 1.5 sürümü, mevcut piyasadaki (örneğin OOTDiffusion, IDM-VTON) modellere kıyasla difüzyon mimarisini en yeni state-of-the-art tasarım paterni olan **FLUX/MMDiT çift akışlı transformer mimarisi** ile baştan yazmış, en kaliteli doku transferini vadeden sistemlerden biridir. Açık kaynak lisansıyla (Apache-2.0) kendi RunPod sisteminize tam uyumlu halde entegre edilebilir bir API hizmeti gibi kullanabilirsiniz.
