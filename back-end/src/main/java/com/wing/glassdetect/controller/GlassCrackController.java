package com.wing.glassdetect.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.model.History;
import com.wing.glassdetect.service.GlassCrackService;
import com.wing.glassdetect.service.HistoryService;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/detect/glass-crack")
public class GlassCrackController {

    @Autowired
    private GlassCrackService glassCrackService;

    @Autowired
    private HistoryService historyService;

    @Autowired
    private ObjectMapper objectMapper;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @Value("${image.storage.path}")
    private String imageStoragePath;

    @PostMapping
    public ResponseEntity<DetectionResult> detectGlassCrack(@RequestParam("userId") Long userId, @RequestParam("images") MultipartFile[] images) {
        String url = algorithmUrl + "/api/detect/glass-crack";
        CompletableFuture<DetectionResult> future = glassCrackService.detect(userId, images, url);

        Path tempImageDir = null;
        try {
            DetectionResult result = future.get(); // 等待异步任务完成

            History history = new History();
            history.setUserId(userId);
            history.setType("crack");
            history.setDate(LocalDateTime.now());
            history.setStatus(result.getStatus());
            history.setTitle(result.getTitle());
            history.setDescription(result.getDescription());

            if (result.getDetails() != null) {
                List<Map<String, String>> detailsList = objectMapper.convertValue(result.getDetails(), new TypeReference<>() {});
                history.setDetails(detailsList);
            }

            // 处理图片存储
            if (result.getImage() != null && !result.getImage().isEmpty()) {
                Path tempImagePath = Paths.get(result.getImage());
                tempImageDir = tempImagePath.getParent();
                String newFileName = UUID.randomUUID().toString() + "_" + tempImagePath.getFileName().toString();
                Path permanentImagePath = Paths.get(imageStoragePath, newFileName);

                Files.createDirectories(permanentImagePath.getParent());
                Files.copy(tempImagePath, permanentImagePath, StandardCopyOption.REPLACE_EXISTING);

                String webImagePath = "/images/" + newFileName;
                history.setImage(webImagePath);
                result.setImage(webImagePath); // 更新返回给前端的路径
            }

            historyService.saveHistory(history);

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            // 处理异常，例如 CompletableFuture 的 ExecutionException
            DetectionResult errorResult = new DetectionResult("error", "检测失败", e.getMessage(), null);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResult);
        } finally {
            // 清理临时文件
            if (tempImageDir != null) {
                FileUtils.deleteTempDir(tempImageDir);
            }
        }
    }
}
