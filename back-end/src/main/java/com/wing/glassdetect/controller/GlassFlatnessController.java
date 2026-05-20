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
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;

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
    public DeferredResult<ResponseEntity<DetectionResult>> detectGlassFlatness(
            @RequestParam("email") String email,
            @RequestParam("left_env") MultipartFile leftEnv,
            @RequestParam("left_mix") MultipartFile leftMix,
            @RequestParam("right_env") MultipartFile rightEnv,
            @RequestParam("right_mix") MultipartFile rightMix) {

        DeferredResult<ResponseEntity<DetectionResult>> deferred = new DeferredResult<>(600000L);

        deferred.onTimeout(() -> deferred.setErrorResult(
                ResponseEntity.status(HttpStatus.REQUEST_TIMEOUT)
                        .body(new DetectionResult("error", "检测超时", "处理超时，请稍后重试", null))));

        MultipartFile[] images = new MultipartFile[]{leftEnv, leftMix, rightEnv, rightMix};
        String[] fieldNames = {"left_env", "left_mix", "right_env", "right_mix"};
        String url = algorithmUrl + "/api/detect/glass-flatness";

        glassFlatnessService.detect(images, fieldNames, url)
                .thenAccept(taskResult -> {
                    DetectionResult result = taskResult.getDetectionResult();
                    Path tempDir = taskResult.getTempDirectory();
                    Path[] tempFiles = taskResult.getTempFiles();
                    persistenceService.normalizeResultImagePath(result);
                    deferred.setResult(ResponseEntity.ok(result));
                    try {
                        persistenceService.persistResult(email, "flatness", result, tempFiles);
                    } catch (Exception ignored) {}
                    if (tempDir != null) {
                        FileUtils.deleteTempDir(tempDir);
                    }
                })
                .exceptionally(ex -> {
                    Throwable cause = ex.getCause() != null ? ex.getCause() : ex;
                    deferred.setResult(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body(new DetectionResult("error", "检测失败", cause.getMessage(), null)));
                    return null;
                });

        return deferred;
    }
}
