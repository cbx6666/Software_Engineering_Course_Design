package com.wing.glassdetect.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.model.History;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class DetectionPersistenceService {

    private final HistoryService historyService;
    private final ObjectMapper objectMapper;
    private final String imageStoragePath;

    @Autowired
    public DetectionPersistenceService(HistoryService historyService, 
                                       ObjectMapper objectMapper, 
                                       @Value("${image.storage.path}") String imageStoragePath) {
        this.historyService = historyService;
        this.objectMapper = objectMapper;
        this.imageStoragePath = imageStoragePath;
    }

    public void persistResult(Long userId, String type, DetectionResult result) throws IOException {
        History history = new History();
        history.setUserId(userId);
        history.setType(type);
        history.setDate(LocalDateTime.now());
        history.setStatus(result.getStatus());
        history.setTitle(result.getTitle());
        history.setDescription(result.getDescription());

        // 根据类型处理不同的数据
        if ("flatness".equals(type) && result.getPointcloud() != null) {
            Map<String, Object> pointcloudMap = objectMapper.convertValue(result.getPointcloud(), new TypeReference<>() {});
            history.setPointcloud(pointcloudMap);
        }

        if (result.getDetails() != null) {
            List<Map<String, String>> detailsList = objectMapper.convertValue(result.getDetails(), new TypeReference<>() {});
            history.setDetails(detailsList);
        }

        // 处理图片存储
        if (result.getImage() != null && !result.getImage().isEmpty()) {
            Path tempImagePath = Paths.get(result.getImage());
            String newFileName = UUID.randomUUID().toString() + "_" + tempImagePath.getFileName().toString();
            Path permanentImagePath = Paths.get(imageStoragePath, newFileName);

            Files.createDirectories(permanentImagePath.getParent());
            Files.copy(tempImagePath, permanentImagePath, StandardCopyOption.REPLACE_EXISTING);

            String webImagePath = "/images/" + newFileName;
            history.setImage(webImagePath);
            result.setImage(webImagePath); // 更新返回给前端的路径
        }

        historyService.saveHistory(history);
    }
}

