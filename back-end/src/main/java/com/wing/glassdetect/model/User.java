package com.wing.glassdetect.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 用户表实体（用于登录/注册）
 *
 */
@TableName("glass_user")
@Data
public class User {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    @TableField("email")
    private String email;

    /** BCrypt 哈希后的密码（禁止 JSON 序列化） */
    @JsonIgnore
    @TableField("password_hash")
    private String passwordHash;

    @TableField(value = "created_at", fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
}
