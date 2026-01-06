package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.dto.DetectionTaskResultDto;
import com.wing.glassdetect.service.DetectionPersistenceService;
import com.wing.glassdetect.service.GlassFlatnessService;
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
@RequestMapping("/api/detect/glass-flatness")
public class GlassFlatnessController {

    @Autowired
    private GlassFlatnessService glassFlatnessService;

    @Autowired
    private DetectionPersistenceService persistenceService;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @PostMapping
    public ResponseEntity<DetectionResult> detectGlassFlatness(@RequestParam("userId") Long userId,
                                                               @RequestParam("left_env") MultipartFile leftEnv,
                                                               @RequestParam("left_mix") MultipartFile leftMix,
                                                               @RequestParam("right_env") MultipartFile rightEnv,
                                                               @RequestParam("right_mix") MultipartFile rightMix) {

        MultipartFile[] images = new MultipartFile[]{leftEnv, leftMix, rightEnv, rightMix};
        String[] fieldNames = {"left_env", "left_mix", "right_env", "right_mix"};
        String url = algorithmUrl + "/api/detect/glass-flatness";

        CompletableFuture<DetectionTaskResultDto> future = glassFlatnessService.detect(userId, images, fieldNames, url);
        Path tempImageDir = null;

        try {
            DetectionTaskResultDto taskResult = future.get(); // 等待异步任务完成
            DetectionResult result = taskResult.getDetectionResult();
            tempImageDir = taskResult.getTempDirectory();
            Path[] tempFiles = taskResult.getTempFiles();

            // 调用服务进行持久化
            persistenceService.persistResult(userId, "flatness", result, tempFiles);

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            DetectionResult errorResult = new DetectionResult("error", "检测或保存失败", e.getMessage(), null);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResult);
        } finally {
            if (tempImageDir != null) {
                FileUtils.deleteTempDir(tempImageDir);
            }
        }
    }
}
