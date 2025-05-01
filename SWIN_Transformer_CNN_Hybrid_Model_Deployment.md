# Production Deployment of the trained model of SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py

Below is a **comprehensive, step-by-step guide** to take the trained `SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py` out of the research notebook into:

1. **A scalable, production‐grade inference service** (e.g. Docker + Kubernetes or serverless)  
2. **An on-device, mobile/edge application** (Android/iOS)



## 1. Server‐Side Production Deployment

### A. Model Export  
1. **TorchScript** (cross-platform, no Python dependency at runtime)  
   ```python
   model.eval()
   scripted = torch.jit.script(model)
   scripted.save('model_scripted.pt')
   ```  
2. **ONNX** (vender-neutral, can target TensorRT, OpenVINO, etc.)  
   ```python
   dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
   torch.onnx.export(
       model, dummy, 'model.onnx',
       input_names=['input'], output_names=['output'],
       dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}},
       opset_version=12
   )
   ```

### B. Containerize with Docker  
1. **`requirements.txt`**  
   ```text
   torch
   torchvision
   timm
   fastapi
   uvicorn
   numpy
   pyyaml
   ```
2. **`Dockerfile`**  
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt
   COPY model_scripted.pt app.py ./
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
   ```

3. **`app.py`** (FastAPI skeleton)  
   ```python
   from fastapi import FastAPI, File, UploadFile
   import uvicorn, io
   from PIL import Image
   import torch
   from torchvision import transforms

   app = FastAPI()
   model = torch.jit.load('model_scripted.pt')
   model.eval()
   preprocess = transforms.Compose([
     transforms.Resize(256), transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
   ])

   @app.post('/predict')
   async def predict(file: UploadFile = File(...)):
       data = await file.read()
       img = Image.open(io.BytesIO(data)).convert('RGB')
       inp = preprocess(img).unsqueeze(0)
       with torch.no_grad():
         logits = model(inp)
         prob = torch.softmax(logits, dim=1)[0,1].item()
       label = 'Precancerous' if prob>0.5 else 'Normal'
       return {'label': label, 'probability': prob}

   if __name__ == '__main__':
       uvicorn.run(app, host='0.0.0.0', port=80)
   ```

### C. Build & Push  
```bash
docker build -t yourrepo/colpo-hybrid:latest .
docker push yourrepo/colpo-hybrid:latest
```

### D. Orchestrate & Monitor  
- **Kubernetes / AWS ECS / GCP Cloud Run**  
  - Deploy the image, set resource requests/limits, enable HPA (autoscaling).  
  - Configure liveness/readiness probes on `/predict`.  
  - Expose via Ingress (TLS), enforce auth.  
- **Monitoring**  
  - Use Prometheus exporters for latency, error‐rate.  
  - Aggregate logs with ELK or Cloud Logging.  



## 2. Edge Deployment (Mobile / Embedded)

### A. Quantize & Convert  
1. **Dynamic Quantization** (reduce model size, minimal accuracy loss)  
   ```python
   import torch.quantization as tq
   quantized = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
   ts = torch.jit.script(quantized)
   ts.save('model_quantized.pt')
   ```
2. **Platform‐Specific**  
   - **Android / PyTorch Mobile**: use `model_quantized.pt` directly.  
   - **iOS / Core ML**: convert via `coremltools`:  
     ```python
     import coremltools as ct
     mlmodel = ct.convert(ts, inputs=[ct.ImageType(name="input", shape=(1,3,224,224))])
     mlmodel.save("Model.mlmodel")
     ```

### B. Mobile App Integration  
#### Android (Java/Kotlin)  
- **Bundle** `model_quantized.pt` in `app/src/main/assets/`.  
- Use [PyTorch Android API](https://pytorch.org/mobile/android/):  
  ```kotlin
  val module = LiteModuleLoader.load(assetFilePath(this,"model_quantized.pt"))
  val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, 
     NormalizationUtils.normalizeRGB([0.485f,0.456f,0.406f],[0.229f,0.224f,0.225f]))
  val output = module.forward(IValue.from(inputTensor)).toTensor()
  val prob = output.dataAsFloatArray[1]
  ```
#### iOS (Swift)  
- Add `Model.mlmodel` to your Xcode project.  
- Use Vision+Core ML for inference:  
  ```swift
  let model = try VNCoreMLModel(for: Model().model)
  let request = VNCoreMLRequest(model: model) { req, _ in
    guard let res = req.results?.first as? VNClassificationObservation else { return }
    print(res.identifier, res.confidence)
  }
  let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
  try handler.perform([request])
  ```

### C. On-Device Explainability  
- **Grad-CAM** is lightweight enough to compute per‐frame:  
  - Extract feature map from last ConvNeXt or Swin stage.  
  - Multiply by prediction gradients, resize to input size.  
  - Overlay heatmap on camera preview.

### D. Test & Benchmark  
- **Profiling**: Android Profiler / Xcode Instruments for latency, memory, FPS.  
- **Accuracy validation**: Compare on‐device outputs with server‐side baseline.  

### E. Privacy & Distribution  
- All inference **offline**; no PHI leaves the device.  
- Distribute via Google Play / App Store with clear privacy statements.



### Note  
This two-pronged deployment ensures you have a **robust, scalable server endpoint** for batch or clinical use, and a **lightweight, privacy-preserving mobile solution** for point-of-care or remote screening. Both follow industry best practices for performance, security, and user experience.


```python

```
