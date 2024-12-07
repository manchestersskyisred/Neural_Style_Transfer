from flask import Flask, request, render_template, send_file
from style_trans import StyleTransfer
import os

app = Flask(__name__)

# 配置上传路径
UPLOAD_FOLDER = './uploads/'
RESULT_FOLDER = os.path.join(UPLOAD_FOLDER, 'results/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/style_trans', methods=['POST'])
def style_transfer():
    content_file = request.files['content_image']
    style_file = request.files['style_image']

    # 保存上传文件
    content_path = os.path.join(UPLOAD_FOLDER, 'content.jpg')
    style_path = os.path.join(UPLOAD_FOLDER, 'style.jpg')
    content_file.save(content_path)
    style_file.save(style_path)

    # 执行风格迁移
    model = StyleTransfer(content_path, style_path, RESULT_FOLDER)
    output_path = model.run(iterations=1000)

    # 返回生成的图片
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)