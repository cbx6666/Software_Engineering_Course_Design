package com.wing.glassdetect.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.file.Path;
import java.util.Map;

public class ApiUtils {

    private static final RestTemplate restTemplate = createRestTemplate();

    private static RestTemplate createRestTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10_000);
        factory.setReadTimeout(300_000);
        return new RestTemplate(factory);
    }

    /**
     * 发送单张图片到算法（Crack 检测用）
     */
    public static DetectionResult postImage(Path tempFile, String url) {
        try {
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("image", new FileSystemResource(tempFile.toFile()));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body);
            ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);

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

    public static DetectionResult postImageWithFieldNames(Path[] tempFiles, String[] fieldNames, String url) {
        try {
            // 构建请求体
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            for (int i = 0; i < tempFiles.length; i++) {
                body.add(fieldNames[i], new FileSystemResource(tempFiles[i].toFile()));
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body);
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
