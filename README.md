# Semaphore-Detection-API 🔥

<details>
<summary> Documentation </summary>


test_url : `https://complete-goose-helping.ngrok-free.app`

Method: 
- POST 

URI: 
- {test_url}/classify_semaphore

Headers:
- Content-Type: multipart/form-data

Request Body: 
- `image` as `file`, must be a valid image, large size


Response:

200

--- 

- Detected image confidence > 0.5

```
{
    "success": true,
    "message": {
        "confidence": 1.0,
        "predicted_class": "B"
    }
}
```

- Detected image confidence < 0.5

```
{
    "success": true,
    "message": {
        "confidence": 0.2,
        "predicted_class": "Not a Semaphore"
    },
}
```

400

--- 
- Invalid file format

```
{
    "success": false,
    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
}

```

*notes: for better classification, image should be taken potrait in camera or image size ratio should be 3:4
</details>
