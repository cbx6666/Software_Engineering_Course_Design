package com.wing.glassdetect.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Value("${image.storage.path}")
    private String imageStoragePath;

    @Value("${frontend.url}")
    private String frontendUrl;

    @Bean
    public CorsFilter corsFilter() {
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        CorsConfiguration config = new CorsConfiguration();

        config.setAllowCredentials(true); // 允许 Cookie
        config.addAllowedOrigin(frontendUrl); // 允许配置文件中的前端地址
        config.addAllowedOrigin("http://localhost:3000"); // 允许本地开发调试
        config.addAllowedHeader("*"); // 允许所有 Header
        config.addAllowedMethod("*"); // 允许所有请求方式 (GET/POST/OPTIONS等)

        source.registerCorsConfiguration("/**", config);
        return new CorsFilter(source);
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 确保路径以斜杠结尾，避免 Windows/Linux 路径解析问题
        String location = imageStoragePath.endsWith("/") ? imageStoragePath : imageStoragePath + "/";
        registry.addResourceHandler("/images/**")
                .addResourceLocations("file:" + location);
    }
}