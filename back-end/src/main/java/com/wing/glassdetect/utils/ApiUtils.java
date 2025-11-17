package com.wing.glassdetect.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.file.Path;
import java.util.Map;

public class ApiUtils {

    private static final RestTemplate restTemplate = new RestTemplate();

    /**
     * 通用方法：发送 MultipartFile 到 FastAPI，并返回 DetectionResult
     */
    public static DetectionResult postImage(Path tempFile, String url) {
        try {
            // 构建请求体
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(tempFile.toFile()));

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // 发送请求
            ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);

            // 解析返回结果
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
