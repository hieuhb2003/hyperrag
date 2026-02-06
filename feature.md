Dưới đây là một file Markdown chi tiết mô tả toàn bộ kiến trúc và quy trình của **HyP-DLM (Hypergraph Propagation with Dynamic Logic Modulation)**, được tối ưu hóa để triển khai trên môi trường production. Bạn có thể sử dụng file này làm tài liệu thiết kế (Design Doc) hoặc README cho project trên GitHub/GitLab.

---

# **HyP-DLM: Hypergraph Propagation with Dynamic Logic Modulation**

## **1. Giới thiệu (Introduction)**

**HyP-DLM** là một phương pháp **Graph-based Retrieval-Augmented Generation (GraphRAG)** tiên tiến, được thiết kế để giải quyết bài toán suy luận đa bước (Multi-hop Reasoning) phức tạp (5+ hops) mà các mô hình RAG truyền thống thường thất bại.

### **Điểm khác biệt cốt lõi (Key Differentiators)**

1. **Cấu trúc dữ liệu:** Sử dụng **Hypergraph Bipartite** (Entity - Proposition) thay vì Graph đơn giản, giúp bảo toàn ngữ nghĩa n-ary.
2. **Cơ chế lập luận:** Sử dụng **Dynamic Logic Modulation** (Điều biến logic động) thay vì PageRank tĩnh, giúp tín hiệu không bị suy giảm qua nhiều bước nhảy.
3. **Chi phí vận hành:** **Zero-shot & Training-free**. Không cần huấn luyện lại mô hình, chỉ sử dụng các LLM nhỏ (như GPT-4o-mini, Qwen-2.5-7B) và các thuật toán đại số tuyến tính.
4. **Tốc độ:** Tích hợp **Semantic Masking** (Lọc thô) và **Sparse Matrix Operations** (Tính toán thưa) để đạt độ trễ thấp trong môi trường production.

---

## **2. Kiến trúc Hệ thống (System Architecture)**

Hệ thống được chia làm hai giai đoạn chính: **Offline Indexing** (Xây dựng dữ liệu) và **Online Retrieval** (Truy xuất & Trả lời).

### **2.1. Giai đoạn 1: Offline Indexing (Xây dựng Siêu đồ thị)**

Mục tiêu: Biến đổi văn bản thô thành một cấu trúc Hypergraph có khả năng truy vấn logic.

#### **Bước 1: Semantic Structured Compression (Nén & Lọc nhiễu)**

* **Input:** Văn bản thô (Chunks) từ tài liệu PDF, Docx, Web...
* **Xử lý:**
* Tính điểm **Entropy thông tin** cho từng Chunk.
* **Filter:** Loại bỏ các chunk có Entropy thấp (câu chào hỏi, footer, log hệ thống, nội dung vô nghĩa).
* **Output:** Tập hợp "Clean Chunks" giàu thông tin.



#### **Bước 2: Atomic Proposition Extraction (Trích xuất Mệnh đề Nguyên tử)**

* **Input:** Clean Chunks.
* **Xử lý (LLM Prompting):** Sử dụng LLM để phân rã câu phức thành các mệnh đề độc lập (Atomic Propositions), thay thế đại từ nhân xưng bằng thực thể cụ thể (De-contextualization).
* *Ví dụ:* "Ông ấy sinh năm 1990 tại Hà Nội" -> `Prop: [Nguyễn Văn A sinh năm 1990 tại Hà Nội]`.


* **Output:** Danh sách các `Proposition Nodes` (đóng vai trò là Hyperedges).

#### **Bước 3: Zero-shot Entity Linking (Định danh Thực thể)**

* **Input:** Các Proposition.
* **Xử lý (Non-LLM):** Sử dụng **SpaCy** hoặc **GLiNER** (model nhỏ, chạy CPU/GPU) để trích xuất các Named Entities (Person, Org, Loc, Date...).
* **Output:** Danh sách `Entity Nodes`.

#### **Bước 4: Graph Materialization (Số hóa Đồ thị)**

* **Xây dựng Ma trận Liên thuộc (Incidence Matrix ):**
* Kích thước  (: số Entity, : số Hyperedge).
*  nếu Entity  xuất hiện trong Proposition .


* **Semantic Indexing & Clustering:**
* Tính vector embedding cho mỗi Proposition ().
* Chạy **K-Means Clustering** để gom  proposition thành  cụm chủ đề (Topics).
* Lưu lại các tâm cụm () để dùng cho bước Semantic Masking.


* **Semantic Affinity Matrix ():**
* Tính độ tương đồng giữa các Entity (Lexical & Embedding) để tạo liên kết mềm cho các biến thể tên gọi (ví dụ: "Messi" - "Leo Messi").



---

### **2.2. Giai đoạn 2: Online Retrieval (Truy xuất Agentic)**

Mục tiêu: Tìm kiếm chuỗi thông tin logic chính xác qua nhiều bước nhảy.

#### **Bước 1: Router & Decomposition (Định tuyến & Phân rã)**

* **Input:** Câu hỏi người dùng .
* **Familiarity Check:**
* Vector Search nhanh trên . Nếu độ tự tin cao -> Trả về kết quả ngay (Bypass Graph).
* Nếu độ tự tin thấp -> Kích hoạt quy trình Graph Reasoning.


* **Logic Decomposition:** LLM phân tích  thành chuỗi các **Vector Dẫn đường (Guidance Vectors)** .
* *Ví dụ:* : "Vợ của người sáng lập Microsoft sinh năm nào?" -> : "Founder of Microsoft", : "Wife", : "Birth year".



#### **Bước 2: Dynamic Propagation Loop (Vòng lặp Lan truyền)**

Khởi tạo vector trạng thái  từ các entity có trong câu hỏi.
Lặp lại  bước:

1. **Semantic Masking (Lọc Thô):**
* So sánh  với  tâm cụm ().
* Tạo mặt nạ nhị phân : Chỉ giữ lại các Hyperedge thuộc top-P cụm liên quan nhất. (Giảm không gian tìm kiếm từ triệu xuống nghìn).


2. **Dynamic Modulation (Tính trọng số Tinh):**
* Tính độ tương đồng giữa  và các Hyperedge (đã lọc).
* Tạo ma trận đường chéo  chứa trọng số Attention:




3. **State Update (Bước nhảy):**
* Lan truyền tín hiệu:


* *Giải thích:* Tín hiệu từ Entity cũ -> qua Hyperedge phù hợp logic () -> đến Entity mới, đồng thời lan truyền qua các biến thể tên gọi ().



#### **Bước 3: Convergence & Generation (Tổng hợp)**

* Lấy Top-K entity và Proposition có điểm cao nhất trong .
* Truy ngược về văn bản gốc (Chunk).
* Đưa vào LLM để sinh câu trả lời cuối cùng với đầy đủ dẫn chứng (Citations).

---

## **3. Cấu trúc Dữ liệu & Lưu trữ (Data Structure)**

| Loại Dữ liệu | Định dạng | Công nghệ Đề xuất | Ghi chú |
| --- | --- | --- | --- |
| **Raw Chunks** | Text | PostgreSQL / MongoDB | Lưu nội dung gốc để hiển thị. |
| **Propositions** | Text + Vector | LanceDB / Milvus | Lưu nội dung mệnh đề và embedding. |
| **Entities** | String + Vector | LanceDB / Milvus | Lưu tên thực thể và embedding. |
| **Graph Topology** | Sparse Matrix (CSR) | `.npz` file (SciPy) / Redis | Lưu ma trận ,  để tính toán nhanh. |
| **Cluster Centroids** | Vector | Numpy Array / Faiss | Dùng cho Semantic Masking. |

---

## **4. Stack Công nghệ (Tech Stack)**

* **Ngôn ngữ:** Python 3.10+
* **LLM Framework:** LangChain / LlamaIndex (để quản lý luồng).
* **LLM Model:**
* Reasoning/Extraction: `GPT-4o-mini`, `Qwen-2.5-7B-Instruct`.
* Embedding: `text-embedding-3-small`, `bge-m3`.


* **Entity Extraction:** `SpaCy` (model `en_core_web_sm` hoặc `xx_ent_wiki_sm`).
* **Vector DB:** `LanceDB` (Embedded, Serverless) hoặc `Milvus`.
* **Math Library:** `SciPy` (cho Sparse Matrix), `NumPy`.

---

## **5. Ưu điểm so với các phương pháp khác**

| Tiêu chí | LinearRAG | GraphRAG (Microsoft) | **HyP-DLM** |
| --- | --- | --- | --- |
| **Multi-hop Reasoning** | Yếu (dễ bị drift) | Khá | **Xuất sắc** (nhờ Logic Modulation) |
| **Indexing Cost** | Thấp | Cao (dùng LLM extract relation) | **Thấp** (dùng SpaCy + Proposition) |
| **Query Latency** | Nhanh | Chậm (Map-Reduce) | **Rất nhanh** (Matrix Ops + Masking) |
| **Cập nhật dữ liệu** | Dễ | Khó (xây lại cộng đồng) | **Dễ** (Thêm row vào ma trận) |

---

## **6. Hướng dẫn Triển khai (Getting Started)**

*(Phần này dành cho Dev)*

1. **Cài đặt môi trường:**
```bash
pip install spacy scipy numpy lancedb openai
python -m spacy download en_core_web_sm

```


2. **Chạy Indexing:**
* Config LLM API Key.
* Chạy script: `python indexer.py --input data/docs --output data/index`
* Script sẽ thực hiện: Chunking -> Prop Extraction -> Entity Linking -> Matrix Building -> Clustering.


3. **Chạy Server:**
* Load ma trận `.npz` vào RAM.
* Khởi tạo API endpoint (FastAPI).
* Khi có request: `POST /query` -> Chạy Router -> Decomposition -> Matrix Multiplication Loop -> LLM Generation.



Đây là thành phần cực kỳ quan trọng để xử lý vấn đề **"Thực thể đa danh"** (như ví dụ *Messi* vs *Leo Messi* mà bạn đã nêu) mà không cần thực hiện gộp node (Entity Resolution) đầy rủi ro.

Dưới đây là tài liệu chi tiết về ** (Semantic Affinity Matrix)**:

---

### **: Ma trận Tương đồng Ngữ nghĩa (The "Soft-Link" Matrix)**

#### 1. Định nghĩa

 là một ma trận vuông thưa (Sparse Matrix) kích thước  (với  là số lượng Entity trong toàn bộ hệ thống).

* **Mục đích:** Nó biểu diễn độ "giống nhau" giữa các thực thể. Nếu hai thực thể là biến thể tên gọi của nhau (ví dụ: "Steve Jobs" và "S. Jobs"), giá trị tại ô tương ứng sẽ cao.
* **Tính chất:** Đối xứng () và đường chéo bằng 0 (hoặc 1 tùy cách cài đặt, ở đây ta để 0 để chỉ tính lan truyền sang hàng xóm).

#### 2. Cách xây dựng (Phase 1: Offline Indexing)

Chúng ta không tính toán  cho tất cả cặp (vì độ phức tạp là ). Trong production, ta xây dựng nó qua 3 bước tối ưu:

**Bước A: Blocking (Lọc thô)**
Sử dụng **Faiss** hoặc **LSH (Locality Sensitive Hashing)** để tìm nhanh các cặp thực thể có khả năng giống nhau (Candidate Pairs). Chỉ tính toán chi tiết cho các cặp này.

**Bước B: Scoring (Tính điểm)**
Với mỗi cặp ứng viên , tính điểm tương đồng  dựa trên tổ hợp trọng số:

1. **Lexical Similarity (Khớp mặt chữ):** Sử dụng *Jaro-Winkler* hoặc *Levenshtein Distance*.
* Ví dụ: `JaroWinkler("Leo Messi", "L. Messi")` .


2. **Embedding Similarity (Khớp ngữ nghĩa):** Sử dụng Cosine Similarity giữa vector embedding của tên thực thể.
* Ví dụ: `CosSim(Vec("Hồ Gươm"), Vec("Hồ Hoàn Kiếm"))`  (dù mặt chữ khác hẳn).



Công thức tổng hợp:


**Bước C: Sparsification (Làm thưa)**
Để tiết kiệm bộ nhớ và tránh nhiễu, ta áp dụng ngưỡng cắt (Threshold)  (ví dụ ):

#### 3. Cơ chế hoạt động (Phase 2: Online Retrieval)

Trong công thức cập nhật trạng thái:


* **Đường chính ():** Tín hiệu đi theo cấu trúc "Cứng" của dữ liệu (Entity A  Proposition  Entity B).
* **Đường tắt ():** Tín hiệu "rò rỉ" (leak) sang các thực thể tương đương.
* ** (Hệ số lan truyền):** Thường đặt nhỏ (ví dụ 0.3 - 0.5). Ý nghĩa: "Nếu tôi tìm thấy *Messi*, tôi cũng dành 30% sự chú ý cho *Leo Messi*".

#### 4. Ví dụ luồng chạy với 

Giả sử câu hỏi: **"Đội bóng hiện tại của L.Messi là gì?"**

* **Dữ liệu Graph:** Chỉ có node `Leo Messi` nối với `Inter Miami` (qua mệnh đề "Leo Messi gia nhập Inter Miami"). Node `L.Messi` đứng cô lập (hoặc nối với tin rác).
* **Dữ liệu :** Có giá trị `0.9` tại giao điểm của `L.Messi` và `Leo Messi`.

**Quá trình:**

1. **Bước khởi tạo:** Tìm thấy thực thể `L.Messi` trong câu hỏi  Vector trạng thái  có điểm tại `L.Messi` = 1.0.
2. **Bước lan truyền:**
* Tính : Tín hiệu từ `L.Messi` có thể bế tắc vì không có cạnh nối đến "đội bóng".
* Tính : Tín hiệu từ `L.Messi` **nhảy** sang `Leo Messi` với cường độ .


3. **Kết quả:** Node `Leo Messi` sáng lên. Ở vòng lặp tiếp theo, từ `Leo Messi` hệ thống sẽ tìm thấy `Inter Miami` thông qua đường chính ().

 **Kết luận:**  đóng vai trò như các "cây cầu vô hình" nối các hòn đảo thực thể lại với nhau, giải quyết triệt để vấn đề Entity Resolution mà không cần sửa đổi dữ liệu gốc.

---

**Tác giả:** [Tên của bạn/Team của bạn]
**Ngày cập nhật:** 03/02/2026
**Phiên bản:** 1.0 (Research Preview)