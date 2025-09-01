DeepFake Tespit Web Uygulaması
Bu proje, yüklenen videoların gerçek (Real) mi yoksa sahte (Fake) mi olduğunu tespit etmek için derin öğrenme modeli kullanan, web tabanlı bir DeepFake analiz sistemidir. Modern ve kullanıcı dostu arayüzü sayesinde, videoları sürükle-bırak yöntemiyle kolayca analiz edebilirsiniz.

Proje, temel model altyapısı olarak Daxitdon/DeepFake-Detection reposunu kullanır ve bu modeli bir Flask sunucusu üzerinden erişilebilir hale getirir.

<br>

🚀 Projenin Öne Çıkan Özellikleri
Modern Web Arayüzü: Temiz, duyarlı ve sürükle-bırak destekli dosya yükleme alanı.

Güçlü Backend: Python ve Flask ile oluşturulmuş, video işleme ve model çıkarımını yöneten bir sunucu.

Derin Öğrenme Modeli: Görüntü sınıflandırmada başarısı kanıtlanmış, önceden eğitilmiş bir ResNeXt50 modeli kullanır.

Kare Bazlı Analiz: Videoyu karelere ayırır ve her bir kare için ayrı bir tahmin yaparak genel bir güven skoru hesaplar.

Otomatik Donanım Tespiti: Sunucunun çalıştığı sistemde GPU (CUDA) varsa otomatik olarak kullanarak analiz sürecini hızlandırır, yoksa CPU üzerinden devam eder.

Kolay Kurulum: requirements.txt dosyası sayesinde bağımlılıkların hızlıca kurulmasını sağlar.

<br>

🛠️ Kullanılan Teknolojiler
Backend: Python, Flask, PyTorch, TorchVision, OpenCV, NumPy

Frontend: HTML5, CSS3, JavaScript

Model: ResNeXt50-32x4d

<br>

⚙️ Nasıl Çalışır?
Uygulamanın çalışma mantığı aşağıdaki adımlardan oluşur:

Video Yükleme: Kullanıcı, web arayüzü üzerinden bir video dosyası seçer veya sürükleyip bırakır.

Sunucuya Gönderme: Video, Flask sunucusundaki /Detect endpoint'ine gönderilir.

Video İşleme: Sunucu, OpenCV kütüphanesini kullanarak videodan belirli sayıda (bu projede 20) kare yakalar.

Veri Ön İşleme: Her bir kare, modelin beklediği formata (boyutlandırma, normalizasyon vb.) getirilir.

Tahmin: Ön işlenmiş kareler, PyTorch ile yüklenen ResNeXt50 modeline tek tek verilir. Model, her karenin "Fake" ve "Real" olma olasılıklarını hesaplar.

Sonuç Hesaplama: Tüm karelerden elde edilen olasılıkların ortalaması alınarak videonun genel sonucu (FAKE veya REAL) ve güven yüzdesi (Confidence) belirlenir.

Arayüze Gönderme: Hesaplanan sonuç, JSON formatında web arayüzüne geri gönderilir ve kullanıcıya gösterilir.

Temizlik: Analiz bittikten sonra yüklenen video dosyası sunucudan otomatik olarak silinir.

<br>

📊 Model Performansı
Model, %87'nin üzerinde bir doğruluk oranına sahiptir. Eğitim sürecindeki performans metrikleri aşağıda görselleştirilmiştir.

Confusion Matrix	Eğitim ve Doğrulama Kaybı (Loss)	Eğitim ve Doğrulama Başarısı (Accuracy)



E-Tablolar'a aktar

Hesaplanan Doğruluk (Accuracy): %87.17 

<br>

🚀 Başlarken
Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

Ön Gereksinimler
Python 3.7+

pip (Python paket yöneticisi)

Kurulum Adımları
Projeyi Klonlayın:

Bash

git clone https://github.com/KULLANICI-ADINIZ/PROJE-ADINIZ.git
cd PROJE-ADINIZ
Sanal Ortam Oluşturun ve Aktifleştirin:
Bu, proje bağımlılıklarını sisteminizden izole etmenizi sağlar.

Bash

# Sanal ortamı oluştur
python -m venv venv

# Windows için aktifleştirme
.\venv\Scripts\activate

# macOS/Linux için aktifleştirme
source venv/bin/activate
Gerekli Kütüphaneleri Yükleyin:

Bash

pip install -r requirements.txt
Önceden Eğitilmiş Modeli İndirin:
Bu proje, çalışmak için bir model dosyasına ihtiyaç duyar.

Model dosyasını (df_model.pt) Daxitdon/DeepFake-Detection reposundan veya uygun bir kaynaktan indirin.

Proje ana dizininde model adında bir klasör oluşturun.

İndirdiğiniz df_model.pt dosyasını bu klasörün içine taşıyın.

Proje yapınız şu şekilde görünmelidir:

.
└── model/
    └── df_model.pt
Uygulamayı Başlatın:

Bash

python server.py
Uygulama başlatıldığında terminalde aşağıdaki gibi bir çıktı göreceksiniz:

Uygulama başlatılıyor — Cihaz: cuda  (veya cpu)
 * Running on http://127.0.0.1:5000
Tarayıcıda Açın:
Web tarayıcınızı açın ve http://127.0.0.1:5000 adresine gidin.

<br>

📂 Proje Yapısı
.
├── model/
│   └── df_model.pt          # Önceden eğitilmiş model dosyası (indirilmeli)
├── static/
│   ├── script.js            # Frontend JavaScript kodu
│   └── style.css            # Arayüz stil dosyası
├── templates/
│   └── index.html           # Ana HTML sayfası
├── Uploaded_Files/        # Yüklenen videoların geçici olarak tutulduğu dizin
│
├── server.py                # Flask sunucusu ve ana uygulama mantığı
├── requirements.txt         # Gerekli Python kütüphaneleri
├── README.md                # Bu döküman
│
└── Confusion Matrix.png     # Performans görselleri
└── ... (diğer görseller)
<br>

📄 Lisans
Bu proje, MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına göz atın.

🙏 Teşekkür
Bu projenin temelini oluşturan derin öğrenme modeli ve eğitim altyapısı için Daxitdon/DeepFake-Detection reposunun sahibine teşekkür ederiz.
