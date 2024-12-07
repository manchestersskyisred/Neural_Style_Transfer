document.addEventListener("DOMContentLoaded", () => {
    const contentInput = document.getElementById("content_image");
    const styleInput = document.getElementById("style_image");
    const resultImage = document.getElementById("resultImage");
    const submitBtn = document.getElementById("submitBtn");

    contentInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById("contentPreview").src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    styleInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById("stylePreview").src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    submitBtn.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("content_image", contentInput.files[0]);
        formData.append("style_image", styleInput.files[0]);

        const response = await fetch("/style_trans", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            resultImage.src = url;

            // 更新下载按钮链接
            const downloadBtn = document.getElementById("downloadBtn");
            downloadBtn.href = url;
            downloadBtn.style.display = "inline-block"; // 显示下载按钮
        } else {
            alert("风格迁移失败，请重试！");
        }
    });
});