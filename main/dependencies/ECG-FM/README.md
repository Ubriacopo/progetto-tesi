We just run it on a docker instance.

```bash
git clone https://huggingface.co/spaces/mystic-cbk/ecg-fm-api
cd ecg-fm-api
docker build -t ecg-fm-api .
docker run -p 7860:7860 ecg-fm-api
# To avoid problems with cache
docker run -p 7860:7860   -v /home/jacopo/AI:/app/.cache/huggingface   ecg-fm-api 

# then POST to http://localhost:7860/analyze
```

