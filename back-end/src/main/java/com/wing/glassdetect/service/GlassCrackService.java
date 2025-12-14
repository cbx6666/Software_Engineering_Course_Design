package com.wing.glassdetect.service;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.utils.ApiUtils;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;

@Service
public class GlassCrackService {

    @Async
    public CompletableFuture<DetectionResult> detect(MultipartFile[] imageFiles, String url) {
        Path[] tempFiles = null;
        try {
            tempFiles = FileUtils.saveTempFile(imageFiles);
            DetectionResult result = ApiUtils.postImage(tempFiles, url);
            return CompletableFuture.completedFuture(result);
        } catch (IOException e) {
            DetectionResult errorResult = new DetectionResult("error", "上传图片失败", e.getMessage(), null);
            return CompletableFuture.completedFuture(errorResult);
        } finally {
            if (tempFiles != null) {
                for (Path p : tempFiles) {
                    FileUtils.deleteTempFile(p);
                }
            }
        }
    }
}
