package com.wing.glassdetect.service;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.dto.DetectionTaskResultDto;
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

    @Async("asyncExecutor")
    public CompletableFuture<DetectionTaskResultDto> detect(Long userId, MultipartFile[] imageFiles, String url) {
        Path[] tempFiles = null;
        Path tempDir = null;
        DetectionResult result = null;
        try {
            // 保存到独立子目录
            tempFiles = FileUtils.saveTempFile(imageFiles);
            tempDir = tempFiles[0].getParent(); // 获取当前请求的临时目录
            result = ApiUtils.postImage(tempFiles, url);
            return CompletableFuture.completedFuture(new DetectionTaskResultDto(result, tempDir, tempFiles));
        } catch (IOException e) {
            result = new DetectionResult("error", "上传图片失败", e.getMessage(), null);
            return CompletableFuture.completedFuture(new DetectionTaskResultDto(result, tempDir, tempFiles));
        }
    }
}
