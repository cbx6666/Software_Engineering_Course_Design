package com.wing.glassdetect.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@TableName(value = "glass_history", autoResultMap = true)
@Data
public class History {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String type; // "crack" or "flatness"

    private LocalDateTime date;

    private String status; // "success", "warning", "error"

    private String title;

    private String description;

    private String image;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> pointcloud;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, String>> details;

    // Foreign key to user table
    private Long userId;
}

