package com.wing.glassdetect.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;

@Service
public class GlassCrackService {

    private final RestTemplate restTemplate = new RestTemplate();

    public DetectionResult detect(MultipartFile imageFile, String url) {
        try {
            // 保存图片到临时目录(放在项目根目录下 /imageUploaded)
            Path tempDir = Paths.get(System.getProperty("user.dir"), "imageUploaded");
            if (!Files.exists(tempDir)) {
                Files.createDirectories(tempDir);
            }

            Path tempFile = tempDir.resolve("upload-" + System.currentTimeMillis() + ".jpg");
            Files.copy(imageFile.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(tempFile.toFile()));
            body.add("glass_id", "GLASS-001");

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);

            // 删除临时文件
            Files.deleteIfExists(tempFile);

            // 解析 FastAPI 返回的 JSON
            Map<String, Object> resultMap = response.getBody();
            return new ObjectMapper().convertValue(resultMap, DetectionResult.class);

        } catch (Exception e) {
            return new DetectionResult(
                    "error",
                    "检测失败",
                    "算法运行异常：" + e.getMessage(),
                    null
            );
        }
    }
}
