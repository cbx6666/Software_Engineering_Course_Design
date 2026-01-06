-- 用户表（登录/注册）
-- 如你要自定义表名/字段名，记得同步修改 com.wing.glassdetect.model.User 上的 @TableName/@TableField

CREATE TABLE IF NOT EXISTS glass_user (
  id BIGINT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at DATETIME NULL,
  UNIQUE KEY uk_glass_user_email (email)
);

