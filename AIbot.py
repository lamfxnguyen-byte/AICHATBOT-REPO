from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_supabase import Supabase
import numpy as np

#---------------------chatbot api-------------------#
load_dotenv()

app = Flask(__name__)
app.config['SUPABASE_URL'] = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
app.config['SUPABASE_KEY'] = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

CORS(app)
supabase = Supabase(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Biến lưu embedding trong RAM
knowledge_embeddings = []

def fetch_and_embed_knowledge():
    """Fetch dữ liệu một lần & tạo embedding"""
    global knowledge_embeddings

    tri_thuc = supabase.client.table('bang_tri_thuc_chatbot').select('id, title, content').execute()
    data = tri_thuc.data if tri_thuc.data else []

    knowledge_embeddings = []  # Reset khi load lại

    for item in data:
        text = item["content"]
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        knowledge_embeddings.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "content": text,
            "embedding": np.array(emb, dtype=float)
        })
    print("✅ Embeddings loaded:", len(knowledge_embeddings))


# --- Hàm tìm kiếm tương tự ---
# --- Hàm tìm kiếm tương tự nâng cấp ---
def search_similar(query, top_k=5, min_score=0.7):
    if not knowledge_embeddings:
        return []

    # Tạo embedding cho câu hỏi
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    query_emb = np.array(query_emb, dtype=float)

    scored = []
    for item in knowledge_embeddings:
        # Cosine similarity
        score = np.dot(query_emb, item["embedding"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(item["embedding"])
        )
        #if score >= min_score:  # ✅ lọc theo ngưỡng tương tự
        scored.append((score, item))

    # Sắp xếp theo điểm cao nhất
    scored.sort(reverse=True, key=lambda x: x[0])

    return [x[1] for x in scored[:top_k]]



@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Tìm tri thức bằng embedding
    matched = search_similar(user_message)

    if matched:
        knowledge_text = "\n\n".join(
            [f"• {item['title']}: {item['content']}" for item in matched]
        )
    else:
        knowledge_text = "Không có tri thức phù hợp."

    system_prompt = f"""
    Bạn là chuyên gia thể chất cho vận động viên.
    Chỉ trả lời dựa vào TRI THỨC DƯỚI ĐÂY:

    {knowledge_text}

    Nếu thông tin không đủ, hãy nói rằng bạn chưa có đủ dữ liệu để trả lời chính xác.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        fetch_and_embed_knowledge()  # Load dữ liệu Supabase chỉ 1 lần khi start server
    app.run(debug=True, port=8080)
