import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

function VideoUpload({ onResults, onError }) {
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("video", file);

    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.error) {
          throw new Error(data.error);
        }
        onResults(data); // Pass results to parent component
        setUploading(false); // Reset uploading state
      })
      .catch((error) => {
        console.error("Error:", error);
        onError(error.message); // Pass error to parent component
        setUploading(false); // Reset uploading state
      });
  }, [onResults, onError]); // Add dependencies to useCallback

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: 'video/*' });

  return (
    <div>
      <div {...getRootProps()} style={{
        border: '2px dashed #ccc',
        padding: '20px',
        margin: '20px 0',
        cursor: 'pointer',
        background: isDragActive ? '#e0e0e0' : '#fff'
      }}>
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the video file here...</p>
        ) : (
          <p>Drag 'n' drop a video file here, or click to select a file</p>
        )}
      </div>
      {uploading && <p>Uploading and processing video...</p>}
    </div>
  );
}

export default VideoUpload;