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
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;

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
    public DeferredResult<ResponseEntity<DetectionResult>> detectGlassCrack(
            @RequestParam("email") String email,
            @RequestParam("images") MultipartFile[] images) {

        DeferredResult<ResponseEntity<DetectionResult>> deferred = new DeferredResult<>(600000L);

        deferred.onTimeout(() -> deferred.setErrorResult(
                ResponseEntity.status(HttpStatus.REQUEST_TIMEOUT)
                        .body(new DetectionResult("error", "检测超时", "处理超时，请稍后重试", null))));

        if (images == null || images.length == 0 || images[0].isEmpty()) {
            deferred.setResult(ResponseEntity.badRequest()
                    .body(new DetectionResult("error", "缺少图片", "请上传至少一张检测图片", null)));
            return deferred;
        }

        String url = algorithmUrl + "/api/detect/glass-crack";

        glassCrackService.detect(images[0], url)
                .thenAccept(taskResult -> {
                    DetectionResult result = taskResult.getDetectionResult();
                    Path tempDir = taskResult.getTempDirectory();
                    Path[] tempFiles = taskResult.getTempFiles();
                    try {
                        persistenceService.persistResult(email, "crack", result, tempFiles);
                    } catch (Exception ignored) {}
                    if (tempDir != null) {
                        FileUtils.deleteTempDir(tempDir);
                    }
                    deferred.setResult(ResponseEntity.ok(result));
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
