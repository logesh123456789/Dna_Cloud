-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Apr 23, 2023 at 10:28 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `dna_cloud`
--

-- --------------------------------------------------------

--
-- Table structure for table `dc_admin`
--

CREATE TABLE `dc_admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dc_admin`
--

INSERT INTO `dc_admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `dc_register`
--

CREATE TABLE `dc_register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `city` varchar(20) NOT NULL,
  `public_key` varchar(20) NOT NULL,
  `private_key` varchar(20) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dc_register`
--

INSERT INTO `dc_register` (`id`, `name`, `mobile`, `email`, `city`, `public_key`, `private_key`, `uname`, `pass`, `rdate`, `status`) VALUES
(1, 'Ramesh', 9638527415, 'ramesh@gmail.com', 'Salem', '6fc42c43', '88ed6f0c', 'ramesh', '123456', '14-04-2023', 1);

-- --------------------------------------------------------

--
-- Table structure for table `dc_share`
--

CREATE TABLE `dc_share` (
  `id` int(11) NOT NULL,
  `fid` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dc_share`
--

INSERT INTO `dc_share` (`id`, `fid`, `uname`, `rdate`) VALUES
(1, 1, 'sivam', '21-04-2023');

-- --------------------------------------------------------

--
-- Table structure for table `dc_user`
--

CREATE TABLE `dc_user` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `owner` varchar(20) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `dob` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `location` varchar(50) NOT NULL,
  `desig` varchar(30) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `secret_key` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dc_user`
--

INSERT INTO `dc_user` (`id`, `name`, `owner`, `gender`, `dob`, `mobile`, `email`, `location`, `desig`, `uname`, `pass`, `secret_key`) VALUES
(1, 'Sivam', 'ramesh', 'Male', '1999-06-05', 9963257524, 'sivam@gmail.com', 'Chennai', 'Software', 'sivam', '1234', '56517f19');

-- --------------------------------------------------------

--
-- Table structure for table `dc_user_files`
--

CREATE TABLE `dc_user_files` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `file_type` varchar(100) NOT NULL,
  `file_content` varchar(100) NOT NULL,
  `upload_file` varchar(100) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `filesize1` double NOT NULL,
  `fastafile` varchar(100) NOT NULL,
  `filesize2` double NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dc_user_files`
--

INSERT INTO `dc_user_files` (`id`, `uname`, `file_type`, `file_content`, `upload_file`, `rdate`, `filesize1`, `fastafile`, `filesize2`) VALUES
(1, 'ramesh', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'data', 'F1tut1.docx', '21-04-2023', 16.591796875, 'CF1tut1.fasta', 5.97265625),
(2, 'ramesh', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'data', 'F2data1.docx', '23-04-2023', 12.970703125, 'CF2data1.fasta', 1.3916015625),
(3, 'ramesh', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'data', 'F3tut1.docx', '23-04-2023', 16.591796875, 'CF3tut1.fasta', 5.97265625);
