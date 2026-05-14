# Stage 1: Build with Maven
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml ./
RUN mvn dependency:go-offline -B -q || true
COPY src src
RUN mvn package -DskipTests -B -q

# Stage 2: Run with JRE
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
RUN mkdir -p /app/runtime/images
EXPOSE 9000
ENTRYPOINT ["java", "-jar", "app.jar"]
