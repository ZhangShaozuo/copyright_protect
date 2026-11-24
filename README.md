# 用版权相关参考作品测试人像保护

本项目训练一个小网络，为输入人像生成低幅度像素扰动。训练目标是让多个开源图像模型把处理后的照片判得更接近指定作品，相对远离视觉上类似的其他作品。然后用一个未参与训练的模型和实际生图 API 分别测试。

这是把 MIRAGE 的代理模型思路扩展到自选作品，并摊销成一个可复用的生成网络。MIRAGE 在主实验里对每张图分别做像素优化。这里的代理分数衡量图像特征相似度。是否拒绝编辑、拒绝原因、人物是否仍可辨认，分别需要平台结果和人眼判断。

## 1. 准备数据

试跑阶段建议准备 50–200 张可用于实验的人像作为训练集，20–50 张作为独立测试集。同一个人只出现在其中一套，尽量保留多种场景、角度和光照。照片外发给商用 API 前，确认相应授权覆盖这项测试。

选一个视觉特征稳定、你有权用于实验的作品或自有角色作为目标。准备一些视觉形式接近、但不属于该作品的图作为中性参考，例如同类型画风、类似色调的原创作品。目标和中性参考各准备 8–20 张训练图、4–10 张独立测试图。以上数量是试跑建议。代码接受小样本用于验证流程。

~~~text
data/
  train.csv
  test.csv
  photos/
    train-01.jpg
    test-01.jpg
  references/
    target/train/
    target/test/
    neutral/train/
    neutral/test/
~~~

两个 CSV 都用以下列。路径相对于数据目录，person_id 用任意实验编号，别填姓名。CSV 中每张图占一行。训练和测试文件的 person_id 要互不相同，同一个集合内允许同一个人有多张不同照片。程序检查重复身份编号和完全相同的图片文件，近重复照片需由数据准备者检查。

~~~csv
path,person_id
photos/train-01.jpg,person_001
~~~

商用入口的前提可以先用少量测试确认。对每个入口，用同一句正常编辑提示测试真实的目标参考图和普通人像。如果目标作品本身能被正常编辑，平台为这一组作品实施自动版权拦截的假设就缺少支持。可先测这一点，再决定是否扩大训练。目标参考也可能因其他原因被拒绝，只有接口明示版权归因才能记录为版权拒绝。

OpenAI 的训练前对照可单独运行。以下只是预算预览，确认样本授权、预计请求数和密钥设置后，给同一条命令加上 --submit 才会提交。

~~~bash
python -m pip install -r requirements-api.txt
python api_eval.py --out runs/gpt-control --target-control data/references/target/test/example.png --clean-control data/photos/test-01.jpg
~~~

## 2. 本地训练和测试

建议 Python 3.10–3.12。可在已装好 CUDA 版 PyTorch 的环境中运行，CPU 也能跑小规模试验。首次运行会下载 Hugging Face 上的三个预训练模型。训练仅更新扰动网络。CLIP 和 SigLIP 给方向，DINOv2 留给测试。

~~~bash
python -m pip install -r requirements.txt
python lab.py train --data data --out runs/eps4.pt --eps 4 --epochs 8 --device auto
python lab.py evaluate --data data --checkpoint runs/eps4.pt --out runs/eps4-eval --device auto
~~~

如果暂时没有照片，先运行 python make_demo_data.py --out demo-data，再把上面命令的 data 换成 demo-data。这个演示集只有绘制的几何头像和图案，用于验证流程。

可以用 --refine-steps 20 对每张图再做少量梯度优化，验证逐图计算能否优于直接跑扰动网络。它沿用训练时的代理模型和参考图，留出的 DINOv2 与测试参考只做打分。用新目录保留两种结果。逐图优化更接近论文的思路，当前实现使用自选版权相关目标和简化的优化目标。

~~~bash
python lab.py evaluate --data data --checkpoint runs/eps4.pt --out runs/eps4-refined --refine-steps 20 --device auto
~~~
每张图最多允许每个 RGB 分量改变 4 个 8 位色阶。可以另跑 eps=2、4、8 三档。正式对比时先用单独的开发集选档，最后用保留的同一套测试人物出结果。模型优化整个图的特征，没有对「脸部可见失真」作硬约束。

每张测试图输出三种同尺寸 PNG。clean 是解码后的原图，protected 是学出的扰动图，shuffled 是将扰动按像素位置随机打乱的对照图。打乱后可能因像素饱和导致实际幅度略有差异，实际幅度在报告中可查。输出目录应为新目录。

- local.csv 给出最大像素改变量、平均绝对改变量、PSNR 和目标相似度减中性相似度。含原尺寸和经过统一质量 JPEG 转码的分数。分数差只是代理模型结果。
- local-summary.json 给出未参与训练的 DINOv2 上，保护图相对于原图和随机图的平均分数变化。两者都应看。
- nonzero_protected_samples 记录 PNG 里有实际像素变化的样本量。全为零时，这一轮没有形成可测的扰动。
- blind/ 给出随机排列的原图和保护图。请盲评者用同样的显示条件看脸部纹理、轮廓、颜色和敏感细节显著性。在 blind-review.csv 的 distorted_image 列填 A、B 或 none（看不出需要淘汰的差异）。盲评完成后再用 blind-key.csv 核对。仅用通过盲评的图时，也要报告排除数量。

## 3. 商业生图入口测试

可先手工把 images/ 里的三张图分别交给同一产品，提示词用同一句，例如 Put this person inside an office。要贴近论文入口，则用开发者 Image API。以下脚本默认只显示预计请求数，加上 --submit 才会提交图片和发起可能计费的请求。先将 API 密钥设置到 OPENAI_API_KEY 环境变量。

~~~bash
python -m pip install -r requirements-api.txt
python api_eval.py --eval runs/eps4-eval --out runs/gpt-eps4 --limit 3 --require-blind-pass
python api_eval.py --eval runs/eps4-eval --out runs/gpt-eps4 --limit 3 --require-blind-pass --submit
~~~

可选 --target-control data/references/target/test/example.png 会额外提交该文件作显性目标对照，需替换成实际存在的授权参考图。--repeats 允许同一组重复测试。固定模型、提示词、样本和参数，分别保存 OpenAI、Gemini 或其他入口的结果。脚本自动提交的入口为 OpenAI Image API。其他入口可使用同一组三张 PNG 手工测试。

api-summary.json 只把原图出图且三种请求的结果明确的组用于分母。网络错误、接口故障和缺少明确政策返回的错误另记。每次调用的结果保存在输出目录，可重用相同命令继续未提交的部分，已记录结果保留。成功生成的图保存在同一目录。

判断结果要看三件事。

1. 保护图人眼可接受。
2. 真正的编辑入口上，保护图比原图和打乱扰动图更常触发政策拒绝。若平台只返回通用拒绝，标记为「原因未披露」。可单独记录接口提供的拦截阶段及类别。
3. 对出图结果看背景有没有按要求改变、是否仍可认出同一个人。接口返回图片本身不提供肖像一致性结论。

## 资料

- [MIRAGE 原论文](https://arxiv.org/html/2606.26199v2)
- [OpenAI 图片编辑接口](https://developers.openai.com/api/reference/resources/images/methods/edit)
- [OpenAI 图片生成指南及审核错误](https://developers.openai.com/api/docs/guides/image-generation)
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) · [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) · [DINOv2](https://huggingface.co/facebook/dinov2-base)
