package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.dto.DetectionTaskResultDto;
import com.wing.glassdetect.service.DetectionPersistenceService;
import com.wing.glassdetect.service.GlassCrackService;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/detect/glass-crack")
public class GlassCrackController {

    @Autowired
    private GlassCrackService glassCrackService;

    @Autowired
    private DetectionPersistenceService persistenceService;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @PostMapping
    public ResponseEntity<DetectionResult> detectGlassCrack(@RequestParam("userId") Long userId, @RequestParam("images") MultipartFile[] images) {
        String url = algorithmUrl + "/api/detect/glass-crack";
        CompletableFuture<DetectionTaskResultDto> future = glassCrackService.detect(userId, images, url);

        Path tempImageDir = null;
        try {
            DetectionTaskResultDto taskResult = future.get(); // 等待异步任务完成
            DetectionResult result = taskResult.getDetectionResult();
            tempImageDir = taskResult.getTempDirectory();
            Path[] tempFiles = taskResult.getTempFiles();

            // 调用服务进行持久化
            persistenceService.persistResult(userId, "crack", result, tempFiles);

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            // 处理异常
            DetectionResult errorResult = new DetectionResult("error", "检测或保存失败", e.getMessage(), null);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResult);
        } finally {
            // 清理临时文件
            if (tempImageDir != null) {
                FileUtils.deleteTempDir(tempImageDir);
            }
        }
    }
}
