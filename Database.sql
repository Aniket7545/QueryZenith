-- MySQL dump 10.13  Distrib 8.0.38, for Win64 (x86_64)
--
-- Host: localhost    Database: datas
-- ------------------------------------------------------
-- Server version	8.0.39

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `employees`
--

DROP TABLE IF EXISTS `employees`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `employees` (
  `employee_id` int NOT NULL AUTO_INCREMENT,
  `first_name` varchar(50) NOT NULL,
  `last_name` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `phone_number` varchar(15) DEFAULT NULL,
  `hire_date` date NOT NULL,
  `job_title` varchar(50) NOT NULL,
  `salary` decimal(10,2) NOT NULL,
  `department` varchar(50) NOT NULL,
  PRIMARY KEY (`employee_id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `employees`
--

LOCK TABLES `employees` WRITE;
/*!40000 ALTER TABLE `employees` DISABLE KEYS */;
INSERT INTO `employees` VALUES (1,'John','Doe','john.doe@example.com','123-456-7890','2023-01-15','Software Engineer',80000.00,'Engineering'),(2,'Jane','Smith','jane.smith@example.com','123-456-7891','2022-07-23','Project Manager',90000.00,'Management'),(3,'Emily','Jones','emily.jones@example.com','123-456-7892','2020-05-18','Data Analyst',70000.00,'Data Science'),(4,'Michael','Brown','michael.brown@example.com','123-456-7893','2021-11-03','HR Specialist',65000.00,'Human Resources'),(5,'Sarah','Davis','sarah.davis@example.com','123-456-7894','2019-02-14','Marketing Manager',75000.00,'Marketing'),(6,'David','Wilson','david.wilson@example.com','123-456-7895','2018-03-23','Sales Manager',80000.00,'Sales'),(7,'Laura','Moore','laura.moore@example.com','123-456-7896','2021-08-07','Business Analyst',72000.00,'Business Analysis'),(8,'Daniel','Taylor','daniel.taylor@example.com','123-456-7897','2020-11-30','Product Manager',85000.00,'Product'),(9,'Karen','Anderson','karen.anderson@example.com','123-456-7898','2017-06-22','UX Designer',68000.00,'Design'),(10,'James','Thomas','james.thomas@example.com','123-456-7899','2019-09-10','QA Engineer',69000.00,'Quality Assurance'),(11,'Patricia','Jackson','patricia.jackson@example.com','123-456-7800','2022-05-20','Technical Writer',64000.00,'Documentation'),(12,'Charles','White','charles.white@example.com','123-456-7801','2023-01-11','DevOps Engineer',78000.00,'Infrastructure'),(13,'Barbara','Harris','barbara.harris@example.com','123-456-7802','2018-04-25','Systems Analyst',71000.00,'Systems'),(14,'Steven','Martinez','steven.martinez@example.com','123-456-7803','2020-10-19','Database Administrator',73000.00,'Database'),(15,'Jennifer','Garcia','jennifer.garcia@example.com','123-456-7804','2019-07-29','Network Engineer',76000.00,'Networking'),(16,'Paul','Clark','paul.clark@example.com','123-456-7805','2021-12-15','Security Specialist',82000.00,'Security'),(17,'Linda','Rodriguez','linda.rodriguez@example.com','123-456-7806','2017-05-04','Web Developer',67000.00,'Development'),(18,'Mark','Lewis','mark.lewis@example.com','123-456-7807','2018-02-28','Mobile Developer',68000.00,'Development'),(19,'Elizabeth','Lee','elizabeth.lee@example.com','123-456-7808','2022-11-01','Cloud Engineer',79000.00,'Cloud'),(20,'Robert','Walker','robert.walker@example.com','123-456-7809','2021-04-18','IT Support Specialist',65000.00,'Support'),(21,'Susan','Hall','susan.hall@example.com','123-456-7810','2020-01-30','AI Researcher',91000.00,'Research'),(22,'Anthony','Allen','anthony.allen@example.com','123-456-7811','2019-03-16','Blockchain Developer',93000.00,'Development'),(23,'Nancy','Young','nancy.young@example.com','123-456-7812','2018-12-09','Game Developer',70000.00,'Development'),(24,'Matthew','King','matthew.king@example.com','123-456-7813','2022-06-07','Embedded Systems Engineer',85000.00,'Engineering'),(25,'Betty','Wright','betty.wright@example.com','123-456-7814','2021-07-21','IT Manager',87000.00,'Management'),(26,'Christopher','Scott','christopher.scott@example.com','123-456-7815','2020-09-14','Site Reliability Engineer',90000.00,'Infrastructure'),(27,'Margaret','Torres','margaret.torres@example.com','123-456-7816','2019-04-10','UI Designer',66000.00,'Design'),(28,'Kenneth','Nguyen','kenneth.nguyen@example.com','123-456-7817','2018-01-18','Technical Support',63000.00,'Support'),(29,'Dorothy','Perez','dorothy.perez@example.com','123-456-7818','2023-02-20','SEO Specialist',72000.00,'Marketing'),(30,'George','Roberts','george.roberts@example.com','123-456-7819','2022-08-14','Digital Marketer',70000.00,'Marketing'),(31,'Lisa','Turner','lisa.turner@example.com','123-456-7820','2021-03-27','Scrum Master',75000.00,'Agile');
/*!40000 ALTER TABLE `employees` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-07-24  9:19:31
