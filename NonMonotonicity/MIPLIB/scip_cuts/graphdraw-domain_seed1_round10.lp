\ SCIP STATISTICS
\   Problem name     : t_graphdraw-domain
\   Variables        : 254 (180 binary, 20 integer, 0 implicit integer, 54 continuous)
\   Constraints      : 865
Minimize
 Obj: +1 t_center_dist_c,Therapist +1 t_center_dist_c,Instructor +86 t_dist_r,Client,CertificateTemplate
      +86 t_dist_r,Client,CourseSession +86 t_dist_r,Client,Instructor +86 t_dist_r,Client,Location
      +86 t_dist_r,Client,User +86 t_dist_r,CertificateTemplate,CourseSession +86 t_dist_r,Course,CourseSession
      +86 t_dist_r,CourseSession,Instructor +86 t_dist_r,Instructor,Location +86 t_dist_r,Clinic,Location
      +86 t_dist_r,Clinic,Therapist +86 t_dist_r,Location,Therapist +86 t_dist_r,Location,CourseSession
      +86 t_dist_r,Therapist,CourseSession +86 t_dist_r,Role,User +86 t_dist_r,User,Instructor +86 t_dist_r,User,Therapist
      +86 t_dist_c,Client,CertificateTemplate +86 t_dist_c,Client,CourseSession +86 t_dist_c,Client,Instructor
      +86 t_dist_c,Client,Location +86 t_dist_c,Client,User +86 t_dist_c,CertificateTemplate,CourseSession
      +86 t_dist_c,Course,CourseSession +86 t_dist_c,CourseSession,Instructor +86 t_dist_c,Instructor,Location
      +86 t_dist_c,Clinic,Location +86 t_dist_c,Clinic,Therapist +86 t_dist_c,Location,Therapist +86 t_dist_c,Location,CourseSession
      +86 t_dist_c,Therapist,CourseSession +86 t_dist_c,Role,User +86 t_dist_c,User,Instructor +86 t_dist_c,User,Therapist
      +1 t_center_dist_r,Client +1 t_center_dist_r,CertificateTemplate +1 t_center_dist_r,Course +1 t_center_dist_r,Role
      +1 t_center_dist_r,User +1 t_center_dist_r,CourseSession +1 t_center_dist_r,Clinic +1 t_center_dist_r,Location
      +1 t_center_dist_r,Therapist +1 t_center_dist_r,Instructor +1 t_center_dist_c,Client +1 t_center_dist_c,CertificateTemplate
      +1 t_center_dist_c,Course +1 t_center_dist_c,Role +1 t_center_dist_c,User +1 t_center_dist_c,CourseSession
      +1 t_center_dist_c,Clinic +1 t_center_dist_c,Location
Subject to
 compDistAxis1_r,Clinic,Location: +1 t_x_r,Location -1 t_x_r,Clinic +1 t_dist_r,Clinic,Location >= -0.5
 compDistAxis2_c,Instructor,Location: -1 t_x_c,Location +1 t_x_c,Instructor +1 t_dist_c,Instructor,Location
  >= -0.5
 choose1_CertificateTemplate,CourseSession: +1 t_zV_r,rt,CertificateTemplate,CourseSession +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,CourseSession +1 t_zV_c,lb,CertificateTemplate,CourseSession = +1
 axisVdifRT_c,Role,Therapist: -53 t_zV_c,rt,Role,Therapist -1 t_x_c,Therapist +1 t_x_c,Role >= -42
 compDistAxis1_c,CourseSession,Instructor: -1 t_x_c,CourseSession +1 t_x_c,Instructor +1 t_dist_c,CourseSession,Instructor
  >= +1
 compDistAxis1_r,CourseSession,Instructor: +1 t_x_r,Instructor -1 t_x_r,CourseSession +1 t_dist_r,CourseSession,Instructor
  >= -0.5
 compDistAxis1_c,User,Instructor: -1 t_x_c,User +1 t_x_c,Instructor +1 t_dist_c,User,Instructor >= -0.5
 compDistAxis2_c,Therapist,CourseSession: +1 t_x_c,Therapist -1 t_x_c,CourseSession +1 t_dist_c,Therapist,CourseSession
  >= +1.5
 axisVdifLB_r,Role,CourseSession: -48 t_zV_r,lb,Role,CourseSession +1 t_x_r,CourseSession -1 t_x_r,Role
  >= -42
 compDistAxis2_c,Clinic,Location: -1 t_x_c,Location +1 t_x_c,Clinic +1 t_dist_c,Clinic,Location >= +0
 centerDistAxis1_r,Course: +1 t_x_r,Course +1 t_center_dist_r,Course >= +19
 compDistAxis2_r,Location,Therapist: -1 t_x_r,Therapist +1 t_x_r,Location +1 t_dist_r,Location,Therapist
  >= +0.5
 axisVdifLB_c,Client,Clinic: -53 t_zV_c,lb,Client,Clinic +1 t_x_c,Clinic -1 t_x_c,Client >= -42
 axisVdifLB_r,Course,CourseSession: -48 t_zV_r,lb,Course,CourseSession +1 t_x_r,CourseSession -1 t_x_r,Course
  >= -42
 axisVdifRT_r,Client,CertificateTemplate: -49 t_zV_r,rt,Client,CertificateTemplate -1 t_x_r,CertificateTemplate
  +1 t_x_r,Client >= -42
 axisVdifRT_r,Location,Instructor: -52 t_zV_r,rt,Location,Instructor -1 t_x_r,Instructor +1 t_x_r,Location
  >= -42
 axisVdifLB_c,Course,Role: -52 t_zV_c,lb,Course,Role +1 t_x_c,Role -1 t_x_c,Course >= -42
 compDistAxis2_c,Location,CourseSession: +1 t_x_c,Location -1 t_x_c,CourseSession +1 t_dist_c,Location,CourseSession
  >= +1.5
 centerDistAxis1_r,CertificateTemplate: +1 t_x_r,CertificateTemplate +1 t_center_dist_r,CertificateTemplate
  >= +18.5
 compDistAxis2_r,Clinic,Location: -1 t_x_r,Location +1 t_x_r,Clinic +1 t_dist_r,Clinic,Location >= +0.5
 choose1_User,Clinic: +1 t_zV_r,rt,User,Clinic +1 t_zV_c,rt,User,Clinic +1 t_zV_r,lb,User,Clinic +1 t_zV_c,lb,User,Clinic
  = +1
 compDistAxis1_c,Client,CertificateTemplate: +1 t_x_c,CertificateTemplate -1 t_x_c,Client +1 t_dist_c,Client,CertificateTemplate
  >= -0.5
 compDistAxis2_c,Client,Instructor: +1 t_x_c,Client -1 t_x_c,Instructor +1 t_dist_c,Client,Instructor
  >= +0.5
 centerDistAxis1_r,Clinic: +1 t_x_r,Clinic +1 t_center_dist_r,Clinic >= +18
 compDistAxis2_c,Role,User: -1 t_x_c,User +1 t_x_c,Role +1 t_dist_c,Role,User >= +0.5
 compDistAxis1_r,Client,CertificateTemplate: +1 t_x_r,CertificateTemplate -1 t_x_r,Client +1 t_dist_r,Client,CertificateTemplate
  >= +1
 axisVdifLB_r,Client,Location: -51 t_zV_r,lb,Client,Location +1 t_x_r,Location -1 t_x_r,Client >= -42
 compDistAxis2_r,Instructor,Location: +1 t_x_r,Instructor -1 t_x_r,Location +1 t_dist_r,Instructor,Location
  >= -0.5
 compDistAxis2_r,Client,CourseSession: -1 t_x_r,CourseSession +1 t_x_r,Client +1 t_dist_r,Client,CourseSession
  >= +0
 DistAxisLB1_r,Client,CertificateTemplate: +8 t_zV_c,rt,Client,CertificateTemplate +8 t_zV_c,lb,Client,CertificateTemplate
  +1 t_dist_r,Client,CertificateTemplate >= +8
 compDistAxis2_c,Course,CourseSession: -1 t_x_c,CourseSession +1 t_x_c,Course +1 t_dist_c,Course,CourseSession
  >= +2
 axisVdifLB_c,CertificateTemplate,Instructor: -54 t_zV_c,lb,CertificateTemplate,Instructor -1 t_x_c,CertificateTemplate
  +1 t_x_c,Instructor >= -42
 triangle2_Client,User,Instructor,c: +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Client,Instructor +1 t_zV_c,lb,User,Instructor
  <= +2
 axisVdifLB_c,Client,Instructor: -53 t_zV_c,lb,Client,Instructor -1 t_x_c,Client +1 t_x_c,Instructor
  >= -42
 axisVdifRT_r,User,Clinic: -50 t_zV_r,rt,User,Clinic -1 t_x_r,Clinic +1 t_x_r,User >= -42
 axisVdifLB_r,Course,Clinic: -48 t_zV_r,lb,Course,Clinic +1 t_x_r,Clinic -1 t_x_r,Course >= -42
 DistAxisLB1_r,Location,CourseSession: +9 t_zV_c,rt,CourseSession,Location +9 t_zV_c,lb,CourseSession,Location
  +1 t_dist_r,Location,CourseSession >= +9
 compDistAxis2_c,Client,CourseSession: -1 t_x_c,CourseSession +1 t_x_c,Client +1 t_dist_c,Client,CourseSession
  >= +1.5
 axisVdifLB_r,CertificateTemplate,Role: -49 t_zV_r,lb,CertificateTemplate,Role +1 t_x_r,Role -1 t_x_r,CertificateTemplate
  >= -42
 compDistAxis1_c,Course,CourseSession: +1 t_x_c,CourseSession -1 t_x_c,Course +1 t_dist_c,Course,CourseSession
  >= -2
 compDistAxis1_r,Therapist,CourseSession: -1 t_x_r,Therapist +1 t_x_r,CourseSession +1 t_dist_r,Therapist,CourseSession
  >= +0.5
 compDistAxis1_r,Instructor,Location: -1 t_x_r,Instructor +1 t_x_r,Location +1 t_dist_r,Instructor,Location
  >= +0.5
 centerDistAxis2_c,Course: -1 t_x_c,Course +1 t_center_dist_c,Course >= -17
 choose1_CertificateTemplate,Role: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,Role
  +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,Role = +1
 compDistAxis2_r,Course,CourseSession: -1 t_x_r,CourseSession +1 t_x_r,Course +1 t_dist_r,Course,CourseSession
  >= +1.5
 centerDistAxis1_r,Client: +1 t_x_r,Client +1 t_center_dist_r,Client >= +17.5
 compDistAxis2_r,Client,Instructor: -1 t_x_r,Instructor +1 t_x_r,Client +1 t_dist_r,Client,Instructor
  >= +0.5
 centerDistAxis2_r,Therapist: -1 t_x_r,Therapist +1 t_center_dist_r,Therapist >= -17
 centerDistAxis1_r,Role: +1 t_x_r,Role +1 t_center_dist_r,Role >= +19
 axisVdifLB_c,User,Instructor: -53 t_zV_c,lb,User,Instructor -1 t_x_c,User +1 t_x_c,Instructor >= -42
 centerDistAxis1_r,CourseSession: +1 t_x_r,CourseSession +1 t_center_dist_r,CourseSession >= +17.5
 compDistAxis2_c,User,Instructor: +1 t_x_c,User -1 t_x_c,Instructor +1 t_dist_c,User,Instructor >= +0.5
 centerDistAxis1_c,Instructor: +1 t_x_c,Instructor +1 t_center_dist_c,Instructor >= +16
 triangle1_Client,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,Client,Location
  +1 t_zV_c,lb,Client,Instructor <= +2
 choose1_Client,Therapist: +1 t_zV_r,rt,Client,Therapist +1 t_zV_c,rt,Client,Therapist +1 t_zV_r,lb,Client,Therapist
  +1 t_zV_c,lb,Client,Therapist = +1
 axisVdifLB_c,Course,Instructor: -52 t_zV_c,lb,Course,Instructor -1 t_x_c,Course +1 t_x_c,Instructor
  >= -42
 centerDistAxis1_c,CertificateTemplate: +1 t_x_c,CertificateTemplate +1 t_center_dist_c,CertificateTemplate
  >= +16
 axisVdifLB_r,Course,Location: -48 t_zV_r,lb,Course,Location +1 t_x_r,Location -1 t_x_r,Course >= -42
 centerDistAxis1_c,Role: +1 t_x_c,Role +1 t_center_dist_c,Role >= +17
 compDistAxis1_c,Clinic,Location: +1 t_x_c,Location -1 t_x_c,Clinic +1 t_dist_c,Clinic,Location >= +0
 compDistAxis1_c,Client,Instructor: -1 t_x_c,Client +1 t_x_c,Instructor +1 t_dist_c,Client,Instructor
  >= -0.5
 compDistAxis1_r,Location,Therapist: +1 t_x_r,Therapist -1 t_x_r,Location +1 t_dist_r,Location,Therapist
  >= -0.5
 axisVdifRT_r,Client,Course: -48 t_zV_r,rt,Client,Course -1 t_x_r,Course +1 t_x_r,Client >= -42
 triangle2_Client,Course,Instructor,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,Instructor +1 t_zV_c,lb,Course,Instructor
  <= +2
 compDistAxis2_r,Client,Location: -1 t_x_r,Location +1 t_x_r,Client +1 t_dist_r,Client,Location >= +0
 compDistAxis2_r,CertificateTemplate,CourseSession: -1 t_x_r,CourseSession +1 t_x_r,CertificateTemplate
  +1 t_dist_r,CertificateTemplate,CourseSession >= +1
 axisVdifLB_r,CourseSession,Location: -51 t_zV_r,lb,CourseSession,Location +1 t_x_r,Location -1 t_x_r,CourseSession
  >= -42
 axisVdifLB_c,Role,Instructor: -52 t_zV_c,lb,Role,Instructor -1 t_x_c,Role +1 t_x_c,Instructor >= -42
 compDistAxis1_c,Location,Therapist: +1 t_x_c,Therapist -1 t_x_c,Location +1 t_dist_c,Location,Therapist
  >= +0
 axisVdifLB_r,CertificateTemplate,CourseSession: -49 t_zV_r,lb,CertificateTemplate,CourseSession +1 t_x_r,CourseSession
  -1 t_x_r,CertificateTemplate >= -42
 triangle1_Client,CertificateTemplate,Course,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Course
  +1 t_zV_c,rt,CertificateTemplate,Course <= +2
 DistAxisLB1_r,CertificateTemplate,CourseSession: +8 t_zV_c,rt,CertificateTemplate,CourseSession +8 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_dist_r,CertificateTemplate,CourseSession >= +8
 centerDistAxis2_r,Location: -1 t_x_r,Location +1 t_center_dist_r,Location >= -17.5
 choose1_Role,Therapist: +1 t_zV_r,rt,Role,Therapist +1 t_zV_c,rt,Role,Therapist +1 t_zV_r,lb,Role,Therapist
  +1 t_zV_c,lb,Role,Therapist = +1
 axisVdifLB_c,User,Clinic: -53 t_zV_c,lb,User,Clinic +1 t_x_c,Clinic -1 t_x_c,User >= -42
 choose1_CourseSession,Location: +1 t_zV_r,rt,CourseSession,Location +1 t_zV_c,rt,CourseSession,Location
  +1 t_zV_r,lb,CourseSession,Location +1 t_zV_c,lb,CourseSession,Location = +1
 compDistAxis1_c,CertificateTemplate,CourseSession: +1 t_x_c,CourseSession -1 t_x_c,CertificateTemplate
  +1 t_dist_c,CertificateTemplate,CourseSession >= -1
 axisVdifLB_c,Client,CertificateTemplate: -53 t_zV_c,lb,Client,CertificateTemplate +1 t_x_c,CertificateTemplate
  -1 t_x_c,Client >= -42
 centerDistAxis2_c,Role: -1 t_x_c,Role +1 t_center_dist_c,Role >= -17
 axisVdifLB_r,CourseSession,Therapist: -51 t_zV_r,lb,CourseSession,Therapist +1 t_x_r,Therapist -1 t_x_r,CourseSession
  >= -42
 centerDistAxis2_c,CourseSession: -1 t_x_c,CourseSession +1 t_center_dist_c,CourseSession >= -15
 centerDistAxis2_c,Clinic: -1 t_x_c,Clinic +1 t_center_dist_c,Clinic >= -16.5
 axisVdifLB_c,Therapist,Instructor: -53 t_zV_c,lb,Therapist,Instructor -1 t_x_c,Therapist +1 t_x_c,Instructor
  >= -42
 compDistAxis1_c,Instructor,Location: +1 t_x_c,Location -1 t_x_c,Instructor +1 t_dist_c,Instructor,Location
  >= +0.5
 centerDistAxis2_c,Instructor: -1 t_x_c,Instructor +1 t_center_dist_c,Instructor >= -16
 compDistAxis1_c,Clinic,Therapist: +1 t_x_c,Therapist -1 t_x_c,Clinic +1 t_dist_c,Clinic,Therapist
  >= +0
 triangle1_Client,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,Client,Location
  +1 t_zV_c,lb,Client,Therapist <= +2
 compDistAxis1_r,Location,CourseSession: -1 t_x_r,Location +1 t_x_r,CourseSession +1 t_dist_r,Location,CourseSession
  >= +0
 compDistAxis2_c,CertificateTemplate,CourseSession: -1 t_x_c,CourseSession +1 t_x_c,CertificateTemplate
  +1 t_dist_c,CertificateTemplate,CourseSession >= +1
 axisVdifRT_r,CourseSession,Instructor: -52 t_zV_r,rt,CourseSession,Instructor -1 t_x_r,Instructor
  +1 t_x_r,CourseSession >= -42
 compDistAxis2_r,Client,CertificateTemplate: -1 t_x_r,CertificateTemplate +1 t_x_r,Client +1 t_dist_r,Client,CertificateTemplate
  >= -1
 axisVdifRT_c,Client,Course: -52 t_zV_c,rt,Client,Course -1 t_x_c,Course +1 t_x_c,Client >= -42
 triangle2_Client,Therapist,Instructor,c: +1 t_zV_c,lb,Client,Therapist +1 t_zV_c,rt,Client,Instructor
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 axisVdifLB_c,CertificateTemplate,Clinic: -54 t_zV_c,lb,CertificateTemplate,Clinic +1 t_x_c,Clinic
  -1 t_x_c,CertificateTemplate >= -42
 compDistAxis2_r,Role,User: -1 t_x_r,User +1 t_x_r,Role +1 t_dist_r,Role,User >= +0.5
 axisVdifLB_c,CertificateTemplate,Role: -54 t_zV_c,lb,CertificateTemplate,Role +1 t_x_c,Role -1 t_x_c,CertificateTemplate
  >= -42
 axisVdifLB_r,Role,Clinic: -48 t_zV_r,lb,Role,Clinic +1 t_x_r,Clinic -1 t_x_r,Role >= -42
 compDistAxis2_r,Clinic,Therapist: -1 t_x_r,Therapist +1 t_x_r,Clinic +1 t_dist_r,Clinic,Therapist
  >= +1
 compDistAxis1_c,Role,User: +1 t_x_c,User -1 t_x_c,Role +1 t_dist_c,Role,User >= -0.5
 choose1_User,CourseSession: +1 t_zV_r,rt,User,CourseSession +1 t_zV_c,rt,User,CourseSession +1 t_zV_r,lb,User,CourseSession
  +1 t_zV_c,lb,User,CourseSession = +1
 choose1_Course,Therapist: +1 t_zV_r,rt,Course,Therapist +1 t_zV_c,rt,Course,Therapist +1 t_zV_r,lb,Course,Therapist
  +1 t_zV_c,lb,Course,Therapist = +1
 DistAxisLB1_c,Client,User: +11 t_zV_r,rt,Client,User +11 t_zV_r,lb,Client,User +1 t_dist_c,Client,User
  >= +11
 DistAxisLB1_c,Role,User: +10.5 t_zV_r,rt,Role,User +10.5 t_zV_r,lb,Role,User +1 t_dist_c,Role,User
  >= +10.5
 axisVdifRT_r,Client,Role: -48 t_zV_r,rt,Client,Role -1 t_x_r,Role +1 t_x_r,Client >= -42
 centerDistAxis2_r,CertificateTemplate: -1 t_x_r,CertificateTemplate +1 t_center_dist_r,CertificateTemplate
  >= -18.5
 triangle2_CertificateTemplate,Course,CourseSession,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,Course,CourseSession <= +2
 compDistAxis2_r,Client,User: -1 t_x_r,User +1 t_x_r,Client +1 t_dist_r,Client,User >= -1
 DistAxisLB1_r,Client,Location: +9 t_zV_c,rt,Client,Location +9 t_zV_c,lb,Client,Location +1 t_dist_r,Client,Location
  >= +9
 choose1_CertificateTemplate,Course: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Course
  +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Course = +1
 choose1_Client,Role: +1 t_zV_r,rt,Client,Role +1 t_zV_c,rt,Client,Role +1 t_zV_r,lb,Client,Role +1 t_zV_c,lb,Client,Role
  = +1
 choose1_Role,User: +1 t_zV_r,rt,Role,User +1 t_zV_c,rt,Role,User +1 t_zV_r,lb,Role,User +1 t_zV_c,lb,Role,User
  = +1
 triangle1_Client,Role,User,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Role,User
  <= +2
 choose1_Client,User: +1 t_zV_r,rt,Client,User +1 t_zV_c,rt,Client,User +1 t_zV_r,lb,Client,User +1 t_zV_c,lb,Client,User
  = +1
 axisVdifRT_c,Client,Location: -53 t_zV_c,rt,Client,Location -1 t_x_c,Location +1 t_x_c,Client >= -42
 compDistAxis2_c,Client,Location: -1 t_x_c,Location +1 t_x_c,Client +1 t_dist_c,Client,Location >= +0
 compDistAxis1_c,Client,Location: +1 t_x_c,Location -1 t_x_c,Client +1 t_dist_c,Client,Location >= +0
 centerDistAxis2_c,Client: -1 t_x_c,Client +1 t_center_dist_c,Client >= -16.5
 axisVdifLB_c,Client,User: -53 t_zV_c,lb,Client,User +1 t_x_c,User -1 t_x_c,Client >= -42
 axisVdifRT_c,Client,Role: -52 t_zV_c,rt,Client,Role -1 t_x_c,Role +1 t_x_c,Client >= -42
 axisVdifRT_c,Client,User: -53 t_zV_c,rt,Client,User -1 t_x_c,User +1 t_x_c,Client >= -42
 axisVdifLB_r,Role,Location: -48 t_zV_r,lb,Role,Location +1 t_x_r,Location -1 t_x_r,Role >= -42
 axisVdifRT_c,Client,Therapist: -53 t_zV_c,rt,Client,Therapist -1 t_x_c,Therapist +1 t_x_c,Client
  >= -42
 compDistAxis1_c,Client,User: +1 t_x_c,User -1 t_x_c,Client +1 t_dist_c,Client,User >= +0
 compDistAxis2_c,Client,CertificateTemplate: -1 t_x_c,CertificateTemplate +1 t_x_c,Client +1 t_dist_c,Client,CertificateTemplate
  >= +0.5
 compDistAxis2_c,Client,User: -1 t_x_c,User +1 t_x_c,Client +1 t_dist_c,Client,User >= +0
 axisVdifRT_c,Role,User: -53 t_zV_c,rt,Role,User -1 t_x_c,User +1 t_x_c,Role >= -42
 DistAxisLB1_c,Location,CourseSession: +12.5 t_zV_r,rt,CourseSession,Location +12.5 t_zV_r,lb,CourseSession,Location
  +1 t_dist_c,Location,CourseSession >= +12.5
 compDistAxis2_r,Location,CourseSession: +1 t_x_r,Location -1 t_x_r,CourseSession +1 t_dist_r,Location,CourseSession
  >= +0
 choose1_Client,Instructor: +1 t_zV_r,rt,Client,Instructor +1 t_zV_c,rt,Client,Instructor +1 t_zV_r,lb,Client,Instructor
  +1 t_zV_c,lb,Client,Instructor = +1
 choose1_CertificateTemplate,Therapist: +1 t_zV_r,rt,CertificateTemplate,Therapist +1 t_zV_c,rt,CertificateTemplate,Therapist
  +1 t_zV_r,lb,CertificateTemplate,Therapist +1 t_zV_c,lb,CertificateTemplate,Therapist = +1
 centerDistAxis1_c,Client: +1 t_x_c,Client +1 t_center_dist_c,Client >= +16.5
 axisVdifRT_c,Location,Therapist: -53 t_zV_c,rt,Location,Therapist -1 t_x_c,Therapist +1 t_x_c,Location
  >= -42
 triangle2_Client,Course,Therapist,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,Therapist +1 t_zV_c,lb,Course,Therapist
  <= +2
 axisVdifRT_c,CertificateTemplate,CourseSession: -56 t_zV_c,rt,CertificateTemplate,CourseSession -1 t_x_c,CourseSession
  +1 t_x_c,CertificateTemplate >= -42
 DistAxisLB1_r,Instructor,Location: +9.5 t_zV_c,lb,Location,Instructor +9.5 t_zV_c,rt,Location,Instructor
  +1 t_dist_r,Instructor,Location >= +9.5
 axisVdifLB_c,CertificateTemplate,Location: -54 t_zV_c,lb,CertificateTemplate,Location +1 t_x_c,Location
  -1 t_x_c,CertificateTemplate >= -42
 centerDistAxis2_c,Location: -1 t_x_c,Location +1 t_center_dist_c,Location >= -16.5
 centerDistAxis1_c,Location: +1 t_x_c,Location +1 t_center_dist_c,Location >= +16.5
 axisVdifRT_c,Client,CertificateTemplate: -54 t_zV_c,rt,Client,CertificateTemplate -1 t_x_c,CertificateTemplate
  +1 t_x_c,Client >= -42
 compDistAxis1_c,Client,CourseSession: +1 t_x_c,CourseSession -1 t_x_c,Client +1 t_dist_c,Client,CourseSession
  >= -1.5
 axisVdifLB_c,CourseSession,Location: -56 t_zV_c,lb,CourseSession,Location +1 t_x_c,Location -1 t_x_c,CourseSession
  >= -42
 axisVdifRT_c,Role,Instructor: -54 t_zV_c,rt,Role,Instructor +1 t_x_c,Role -1 t_x_c,Instructor >= -42
 axisVdifLB_r,Client,CourseSession: -51 t_zV_r,lb,Client,CourseSession +1 t_x_r,CourseSession -1 t_x_r,Client
  >= -42
 axisVdifLB_r,Role,Therapist: -48 t_zV_r,lb,Role,Therapist +1 t_x_r,Therapist -1 t_x_r,Role >= -42
 DistSumLB_Role,User: +1 t_dist_r,Role,User +1 t_dist_c,Role,User >= +6.5
 DistAxisLB1_r,Role,User: +6.5 t_zV_c,rt,Role,User +6.5 t_zV_c,lb,Role,User +1 t_dist_r,Role,User
  >= +6.5
 axisVdifRT_r,Clinic,Location: -51 t_zV_r,rt,Clinic,Location -1 t_x_r,Location +1 t_x_r,Clinic >= -42
 triangle1_Client,Role,Instructor,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,Instructor +1 t_zV_c,rt,Role,Instructor
  <= +2
 centerDistAxis2_r,Role: -1 t_x_r,Role +1 t_center_dist_r,Role >= -19
 centerDistAxis1_r,Instructor: +1 t_x_r,Instructor +1 t_center_dist_r,Instructor >= +17
 choose1_Role,CourseSession: +1 t_zV_r,rt,Role,CourseSession +1 t_zV_c,rt,Role,CourseSession +1 t_zV_r,lb,Role,CourseSession
  +1 t_zV_c,lb,Role,CourseSession = +1
 choose1_Course,Instructor: +1 t_zV_r,rt,Course,Instructor +1 t_zV_c,rt,Course,Instructor +1 t_zV_r,lb,Course,Instructor
  +1 t_zV_c,lb,Course,Instructor = +1
 triangle2_CertificateTemplate,Role,CourseSession,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,Role,CourseSession <= +2
 axisVdifLB_r,Client,Course: -51 t_zV_r,lb,Client,Course +1 t_x_r,Course -1 t_x_r,Client >= -42
 centerDistAxis2_r,User: -1 t_x_r,User +1 t_center_dist_r,User >= -18.5
 axisVdifRT_c,Clinic,Location: -53 t_zV_c,rt,Clinic,Location -1 t_x_c,Location +1 t_x_c,Clinic >= -42
 DistSumLB_Client,Location: +1 t_dist_r,Client,Location +1 t_dist_c,Client,Location >= +9
 DistAxisLB1_r,Client,Instructor: +9.5 t_zV_c,rt,Client,Instructor +9.5 t_zV_c,lb,Client,Instructor
  +1 t_dist_r,Client,Instructor >= +9.5
 DistAxisLB1_c,Clinic,Location: +11 t_zV_r,lb,Clinic,Location +11 t_zV_r,rt,Clinic,Location +1 t_dist_c,Clinic,Location
  >= +11
 choose1_Clinic,Location: +1 t_zV_c,lb,Clinic,Location +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,rt,Clinic,Location
  +1 t_zV_c,rt,Clinic,Location = +1
 axisVdifRT_c,CertificateTemplate,Course: -52 t_zV_c,rt,CertificateTemplate,Course -1 t_x_c,Course
  +1 t_x_c,CertificateTemplate >= -42
 centerDistAxis1_c,Course: +1 t_x_c,Course +1 t_center_dist_c,Course >= +17
 triangle1_Client,CertificateTemplate,CourseSession,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,CourseSession
  +1 t_zV_r,rt,CertificateTemplate,CourseSession <= +2
 triangle2_Client,CertificateTemplate,Instructor,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Instructor
  +1 t_zV_c,lb,CertificateTemplate,Instructor <= +2
 choose1_Role,Instructor: +1 t_zV_r,rt,Role,Instructor +1 t_zV_c,rt,Role,Instructor +1 t_zV_r,lb,Role,Instructor
  +1 t_zV_c,lb,Role,Instructor = +1
 axisVdifRT_r,Course,Instructor: -52 t_zV_r,rt,Course,Instructor -1 t_x_r,Instructor +1 t_x_r,Course
  >= -42
 triangle1_Client,Role,CourseSession,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,CourseSession
  +1 t_zV_r,rt,Role,CourseSession <= +2
 axisVdifRT_r,CertificateTemplate,Role: -48 t_zV_r,rt,CertificateTemplate,Role -1 t_x_r,Role +1 t_x_r,CertificateTemplate
  >= -42
 axisVdifLB_r,CertificateTemplate,Course: -49 t_zV_r,lb,CertificateTemplate,Course +1 t_x_r,Course
  -1 t_x_r,CertificateTemplate >= -42
 axisVdifRT_c,Location,Instructor: -54 t_zV_c,rt,Location,Instructor +1 t_x_c,Location -1 t_x_c,Instructor
  >= -42
 choose1_Location,Instructor: +1 t_zV_c,lb,Location,Instructor +1 t_zV_r,lb,Location,Instructor +1 t_zV_c,rt,Location,Instructor
  +1 t_zV_r,rt,Location,Instructor = +1
 centerDistAxis2_c,CertificateTemplate: -1 t_x_c,CertificateTemplate +1 t_center_dist_c,CertificateTemplate
  >= -16
 axisVdifLB_c,Course,Therapist: -52 t_zV_c,lb,Course,Therapist +1 t_x_c,Therapist -1 t_x_c,Course
  >= -42
 triangle1_Client,Role,Therapist,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,Therapist +1 t_zV_c,rt,Role,Therapist
  <= +2
 axisVdifLB_c,Client,Location: -53 t_zV_c,lb,Client,Location +1 t_x_c,Location -1 t_x_c,Client >= -42
 DistSumLB_CertificateTemplate,CourseSession: +1 t_dist_r,CertificateTemplate,CourseSession +1 t_dist_c,CertificateTemplate,CourseSession
  >= +8
 DistAxisLB1_c,CertificateTemplate,CourseSession: +13 t_zV_r,rt,CertificateTemplate,CourseSession +13 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_dist_c,CertificateTemplate,CourseSession >= +13
 axisVdifLB_c,CourseSession,Clinic: -56 t_zV_c,lb,CourseSession,Clinic +1 t_x_c,Clinic -1 t_x_c,CourseSession
  >= -42
 compDistAxis1_r,Clinic,Therapist: +1 t_x_r,Therapist -1 t_x_r,Clinic +1 t_dist_r,Clinic,Therapist
  >= -1
 DistSumLB_Client,CertificateTemplate: +1 t_dist_r,Client,CertificateTemplate +1 t_dist_c,Client,CertificateTemplate
  >= +8
 centerDistAxis1_r,Location: +1 t_x_r,Location +1 t_center_dist_r,Location >= +17.5
 triangle2_Client,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,Client,Location
  +1 t_zV_c,rt,Client,Instructor <= +2
 axisVdifLB_r,CertificateTemplate,User: -49 t_zV_r,lb,CertificateTemplate,User +1 t_x_r,User -1 t_x_r,CertificateTemplate
  >= -42
 axisVdifRT_r,CertificateTemplate,Course: -48 t_zV_r,rt,CertificateTemplate,Course -1 t_x_r,Course
  +1 t_x_r,CertificateTemplate >= -42
 compDistAxis1_r,Client,Location: +1 t_x_r,Location -1 t_x_r,Client +1 t_dist_r,Client,Location >= +0
 DistAxisLB1_c,Instructor,Location: +11.5 t_zV_r,lb,Location,Instructor +11.5 t_zV_r,rt,Location,Instructor
  +1 t_dist_c,Instructor,Location >= +11.5
 axisVdifRT_r,CertificateTemplate,CourseSession: -51 t_zV_r,rt,CertificateTemplate,CourseSession -1 t_x_r,CourseSession
  +1 t_x_r,CertificateTemplate >= -42
 compDistAxis1_r,CertificateTemplate,CourseSession: +1 t_x_r,CourseSession -1 t_x_r,CertificateTemplate
  +1 t_dist_r,CertificateTemplate,CourseSession >= -1
 DistAxisLB1_r,Location,Therapist: +9.5 t_zV_c,lb,Location,Therapist +9.5 t_zV_c,rt,Location,Therapist
  +1 t_dist_r,Location,Therapist >= +9.5
 DistSumLB_Location,Therapist: +1 t_dist_r,Location,Therapist +1 t_dist_c,Location,Therapist >= +9.5
 DistSumLB_Location,CourseSession: +1 t_dist_r,Location,CourseSession +1 t_dist_c,Location,CourseSession
  >= +9
 DistAxisLB1_c,Client,CertificateTemplate: +11.5 t_zV_r,rt,Client,CertificateTemplate +11.5 t_zV_r,lb,Client,CertificateTemplate
  +1 t_dist_c,Client,CertificateTemplate >= +11.5
 choose1_Therapist,Instructor: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_c,rt,Therapist,Instructor
  +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_c,lb,Therapist,Instructor = +1
 triangle2_Client,CertificateTemplate,Role,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Role
  +1 t_zV_c,lb,CertificateTemplate,Role <= +2
 axisVdifLB_r,Location,Therapist: -51 t_zV_r,lb,Location,Therapist +1 t_x_r,Therapist -1 t_x_r,Location
  >= -42
 axisVdifLB_c,Location,Instructor: -53 t_zV_c,lb,Location,Instructor -1 t_x_c,Location +1 t_x_c,Instructor
  >= -42
 DistAxisLB1_c,User,Instructor: +11.5 t_zV_r,rt,User,Instructor +11.5 t_zV_r,lb,User,Instructor +1 t_dist_c,User,Instructor
  >= +11.5
 compDistAxis1_r,User,Instructor: +1 t_x_r,Instructor -1 t_x_r,User +1 t_dist_r,User,Instructor >= -1.5
 CenterDistSumLB: +1 t_center_dist_c,Therapist +1 t_center_dist_c,Instructor +1 t_center_dist_r,Client
  +1 t_center_dist_r,CertificateTemplate +1 t_center_dist_r,Course +1 t_center_dist_r,Role +1 t_center_dist_r,User
  +1 t_center_dist_r,CourseSession +1 t_center_dist_r,Clinic +1 t_center_dist_r,Location +1 t_center_dist_r,Therapist
  +1 t_center_dist_r,Instructor +1 t_center_dist_c,Client +1 t_center_dist_c,CertificateTemplate +1 t_center_dist_c,Course
  +1 t_center_dist_c,Role +1 t_center_dist_c,User +1 t_center_dist_c,CourseSession +1 t_center_dist_c,Clinic
  +1 t_center_dist_c,Location >= +30
 axisVdifRT_r,Role,CourseSession: -51 t_zV_r,rt,Role,CourseSession -1 t_x_r,CourseSession +1 t_x_r,Role
  >= -42
 axisVdifLB_r,CertificateTemplate,Therapist: -49 t_zV_r,lb,CertificateTemplate,Therapist +1 t_x_r,Therapist
  -1 t_x_r,CertificateTemplate >= -42
 axisVdifRT_r,Clinic,Instructor: -52 t_zV_r,rt,Clinic,Instructor -1 t_x_r,Instructor +1 t_x_r,Clinic
  >= -42
 choose1_Client,Course: +1 t_zV_r,rt,Client,Course +1 t_zV_c,rt,Client,Course +1 t_zV_r,lb,Client,Course
  +1 t_zV_c,lb,Client,Course = +1
 axisVdifRT_r,User,Therapist: -52 t_zV_r,rt,User,Therapist -1 t_x_r,Therapist +1 t_x_r,User >= -42
 triangle1_Client,CertificateTemplate,Course,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Course
  +1 t_zV_r,rt,CertificateTemplate,Course <= +2
 DistSumLB_Instructor,Location: +1 t_dist_r,Instructor,Location +1 t_dist_c,Instructor,Location >= +9.5
 axisVdifRT_r,Role,Instructor: -52 t_zV_r,rt,Role,Instructor -1 t_x_r,Instructor +1 t_x_r,Role >= -42
 compDistAxis2_c,Location,Therapist: -1 t_x_c,Therapist +1 t_x_c,Location +1 t_dist_c,Location,Therapist
  >= +0
 axisVdifRT_r,User,Location: -51 t_zV_r,rt,User,Location -1 t_x_r,Location +1 t_x_r,User >= -42
 axisVdifLB_c,CertificateTemplate,CourseSession: -54 t_zV_c,lb,CertificateTemplate,CourseSession +1 t_x_c,CourseSession
  -1 t_x_c,CertificateTemplate >= -42
 compDistAxis1_r,Client,Instructor: +1 t_x_r,Instructor -1 t_x_r,Client +1 t_dist_r,Client,Instructor
  >= -0.5
 axisVdifRT_c,Role,CourseSession: -56 t_zV_c,rt,Role,CourseSession -1 t_x_c,CourseSession +1 t_x_c,Role
  >= -42
 compDistAxis1_c,Location,CourseSession: -1 t_x_c,Location +1 t_x_c,CourseSession +1 t_dist_c,Location,CourseSession
  >= -1.5
 triangle2_Client,CertificateTemplate,Location,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Location
  +1 t_zV_c,lb,CertificateTemplate,Location <= +2
 centerDistAxis1_c,CourseSession: +1 t_x_c,CourseSession +1 t_center_dist_c,CourseSession >= +15
 choose1_User,Instructor: +1 t_zV_r,rt,User,Instructor +1 t_zV_c,rt,User,Instructor +1 t_zV_r,lb,User,Instructor
  +1 t_zV_c,lb,User,Instructor = +1
 choose1_Clinic,Instructor: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_c,rt,Clinic,Instructor
  +1 t_zV_r,rt,Clinic,Instructor = +1
 axisVdifLB_c,CertificateTemplate,Therapist: -54 t_zV_c,lb,CertificateTemplate,Therapist +1 t_x_c,Therapist
  -1 t_x_c,CertificateTemplate >= -42
 axisVdifLB_c,Client,Therapist: -53 t_zV_c,lb,Client,Therapist +1 t_x_c,Therapist -1 t_x_c,Client
  >= -42
 compDistAxis2_c,CourseSession,Instructor: +1 t_x_c,CourseSession -1 t_x_c,Instructor +1 t_dist_c,CourseSession,Instructor
  >= -1
 compDistAxis1_c,Therapist,CourseSession: -1 t_x_c,Therapist +1 t_x_c,CourseSession +1 t_dist_c,Therapist,CourseSession
  >= -1.5
 triangle1_CertificateTemplate,Role,CourseSession,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,Role,CourseSession <= +2
 axisVdifRT_c,Client,Instructor: -54 t_zV_c,rt,Client,Instructor +1 t_x_c,Client -1 t_x_c,Instructor
  >= -42
 axisVdifRT_c,User,CourseSession: -56 t_zV_c,rt,User,CourseSession -1 t_x_c,CourseSession +1 t_x_c,User
  >= -42
 triangle2_Client,CertificateTemplate,Therapist,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Therapist
  +1 t_zV_c,lb,CertificateTemplate,Therapist <= +2
 choose1_Client,CertificateTemplate: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_c,rt,Client,CertificateTemplate
  +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_c,lb,Client,CertificateTemplate = +1
 axisVdifLB_r,Clinic,Location: -50 t_zV_r,lb,Clinic,Location +1 t_x_r,Location -1 t_x_r,Clinic >= -42
 compDistAxis1_r,User,Therapist: +1 t_x_r,Therapist -1 t_x_r,User +1 t_dist_r,User,Therapist >= -1.5
 centerDistAxis2_r,Clinic: -1 t_x_r,Clinic +1 t_center_dist_r,Clinic >= -18
 axisVdifLB_r,Course,Instructor: -48 t_zV_r,lb,Course,Instructor +1 t_x_r,Instructor -1 t_x_r,Course
  >= -42
 DistSumLB_Clinic,Location: +1 t_dist_r,Clinic,Location +1 t_dist_c,Clinic,Location >= +8.5
 DistAxisLB1_r,Clinic,Location: +8.5 t_zV_c,lb,Clinic,Location +8.5 t_zV_c,rt,Clinic,Location +1 t_dist_r,Clinic,Location
  >= +8.5
 axisVdifLB_c,Client,Course: -53 t_zV_c,lb,Client,Course +1 t_x_c,Course -1 t_x_c,Client >= -42
 axisVdifRT_c,Clinic,Instructor: -54 t_zV_c,rt,Clinic,Instructor +1 t_x_c,Clinic -1 t_x_c,Instructor
  >= -42
 choose1_Location,Therapist: +1 t_zV_c,lb,Location,Therapist +1 t_zV_r,lb,Location,Therapist +1 t_zV_c,rt,Location,Therapist
  +1 t_zV_r,rt,Location,Therapist = +1
 DistAxisLB1_c,Location,Therapist: +11 t_zV_r,lb,Location,Therapist +11 t_zV_r,rt,Location,Therapist
  +1 t_dist_c,Location,Therapist >= +11
 triangle2_Client,Course,Role,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Course,Role
  <= +2
 DistAxisLB1_c,Client,Location: +11 t_zV_r,rt,Client,Location +11 t_zV_r,lb,Client,Location +1 t_dist_c,Client,Location
  >= +11
 choose1_Client,Location: +1 t_zV_r,rt,Client,Location +1 t_zV_c,rt,Client,Location +1 t_zV_r,lb,Client,Location
  +1 t_zV_c,lb,Client,Location = +1
 triangle2_Client,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,Client,Location
  <= +2
 triangle2_Clinic,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,rt,Clinic,Instructor
  +1 t_zV_c,lb,Clinic,Location <= +2
 axisVdifRT_c,Role,Location: -53 t_zV_c,rt,Role,Location -1 t_x_c,Location +1 t_x_c,Role >= -42
 triangle1_Client,Role,Location,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,Location +1 t_zV_c,rt,Role,Location
  <= +2
 triangle1_Role,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,Role,Location
  +1 t_zV_c,lb,Role,Instructor <= +2
 axisVdifLB_c,Role,Location: -52 t_zV_c,lb,Role,Location +1 t_x_c,Location -1 t_x_c,Role >= -42
 triangle2_Role,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,Role,Location
  +1 t_zV_c,rt,Role,Instructor <= +2
 axisVdifLB_r,Role,User: -48 t_zV_r,lb,Role,User +1 t_x_r,User -1 t_x_r,Role >= -42
 axisVdifRT_r,Role,Location: -51 t_zV_r,rt,Role,Location -1 t_x_r,Location +1 t_x_r,Role >= -42
 triangle1_Client,Role,Location,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,Location +1 t_zV_r,rt,Role,Location
  <= +2
 triangle2_Role,CourseSession,Location,r: +1 t_zV_r,lb,Role,CourseSession +1 t_zV_r,rt,Role,Location
  +1 t_zV_r,lb,CourseSession,Location <= +2
 triangle2_Role,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,Role,Clinic +1 t_zV_r,rt,Role,Location
  <= +2
 choose1_Client,Clinic: +1 t_zV_r,rt,Client,Clinic +1 t_zV_c,rt,Client,Clinic +1 t_zV_r,lb,Client,Clinic
  +1 t_zV_c,lb,Client,Clinic = +1
 triangle1_Client,Clinic,Location,c: +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,Client,Location +1 t_zV_c,rt,Clinic,Location
  <= +2
 triangle1_Client,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,Client,Therapist
  <= +2
 triangle1_Client,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,Client,Instructor
  <= +2
 triangle2_Client,CertificateTemplate,Clinic,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Clinic
  +1 t_zV_c,lb,CertificateTemplate,Clinic <= +2
 triangle2_Client,User,Clinic,c: +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,User,Clinic
  <= +2
 triangle1_Clinic,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,lb,Clinic,Instructor
  +1 t_zV_c,rt,Clinic,Location <= +2
 triangle2_Client,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,Client,Instructor
  <= +2
 choose1_Role,Location: +1 t_zV_r,rt,Role,Location +1 t_zV_c,rt,Role,Location +1 t_zV_r,lb,Role,Location
  +1 t_zV_c,lb,Role,Location = +1
 axisVdifLB_c,Clinic,Location: -53 t_zV_c,lb,Clinic,Location +1 t_x_c,Location -1 t_x_c,Clinic >= -42
 axisVdifLB_c,Clinic,Instructor: -53 t_zV_c,lb,Clinic,Instructor -1 t_x_c,Clinic +1 t_x_c,Instructor
  >= -42
 axisVdifRT_c,Client,Clinic: -53 t_zV_c,rt,Client,Clinic -1 t_x_c,Clinic +1 t_x_c,Client >= -42
 axisVdifLB_c,Course,User: -52 t_zV_c,lb,Course,User +1 t_x_c,User -1 t_x_c,Course >= -42
 triangle2_Client,Course,User,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Course,User
  <= +2
 centerDistAxis1_c,Clinic: +1 t_x_c,Clinic +1 t_center_dist_c,Clinic >= +16.5
 choose1_Course,User: +1 t_zV_r,rt,Course,User +1 t_zV_c,rt,Course,User +1 t_zV_r,lb,Course,User +1 t_zV_c,lb,Course,User
  = +1
 DistAxisLB1_r,Clinic,Therapist: +9 t_zV_c,lb,Clinic,Therapist +9 t_zV_c,rt,Clinic,Therapist +1 t_dist_r,Clinic,Therapist
  >= +9
 DistSumLB_Clinic,Therapist: +1 t_dist_r,Clinic,Therapist +1 t_dist_c,Clinic,Therapist >= +9
 compDistAxis2_c,Clinic,Therapist: -1 t_x_c,Therapist +1 t_x_c,Clinic +1 t_dist_c,Clinic,Therapist
  >= +0
 axisVdifRT_c,Clinic,Therapist: -53 t_zV_c,rt,Clinic,Therapist -1 t_x_c,Therapist +1 t_x_c,Clinic
  >= -42
 triangle2_Clinic,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,rt,Clinic,Therapist
  +1 t_zV_r,lb,Clinic,Location <= +2
 DistAxisLB1_c,Clinic,Therapist: +11 t_zV_r,lb,Clinic,Therapist +11 t_zV_r,rt,Clinic,Therapist +1 t_dist_c,Clinic,Therapist
  >= +11
 axisVdifRT_r,Clinic,Therapist: -52 t_zV_r,rt,Clinic,Therapist -1 t_x_r,Therapist +1 t_x_r,Clinic
  >= -42
 choose1_Clinic,Therapist: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_c,rt,Clinic,Therapist
  +1 t_zV_r,rt,Clinic,Therapist = +1
 axisVdifLB_r,Clinic,Therapist: -50 t_zV_r,lb,Clinic,Therapist +1 t_x_r,Therapist -1 t_x_r,Clinic
  >= -42
 triangle1_Role,User,Clinic,r: +1 t_zV_r,rt,Role,User +1 t_zV_r,lb,Role,Clinic +1 t_zV_r,rt,User,Clinic
  <= +2
 triangle1_Role,User,Location,r: +1 t_zV_r,rt,Role,User +1 t_zV_r,lb,Role,Location +1 t_zV_r,rt,User,Location
  <= +2
 triangle1_Role,User,Therapist,r: +1 t_zV_r,rt,Role,User +1 t_zV_r,lb,Role,Therapist +1 t_zV_r,rt,User,Therapist
  <= +2
 triangle1_CertificateTemplate,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,CertificateTemplate,Therapist <= +2
 triangle2_CertificateTemplate,Course,Clinic,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,Course,Clinic <= +2
 triangle2_CertificateTemplate,Role,Clinic,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,Role,Clinic <= +2
 axisVdifLB_r,CertificateTemplate,Clinic: -49 t_zV_r,lb,CertificateTemplate,Clinic +1 t_x_r,Clinic
  -1 t_x_r,CertificateTemplate >= -42
 axisVdifRT_r,CertificateTemplate,Clinic: -50 t_zV_r,rt,CertificateTemplate,Clinic -1 t_x_r,Clinic
  +1 t_x_r,CertificateTemplate >= -42
 triangle1_Client,Role,User,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Role,User
  <= +2
 triangle1_Client,Role,Therapist,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,Therapist +1 t_zV_r,rt,Role,Therapist
  <= +2
 triangle1_CertificateTemplate,Role,Therapist,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,Therapist
  +1 t_zV_r,rt,Role,Therapist <= +2
 triangle2_Role,CourseSession,Therapist,r: +1 t_zV_r,lb,Role,CourseSession +1 t_zV_r,rt,Role,Therapist
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 triangle2_Role,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,Role,Clinic +1 t_zV_r,rt,Role,Therapist
  <= +2
 triangle2_Role,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,Role,Location +1 t_zV_r,rt,Role,Therapist
  <= +2
 triangle1_CertificateTemplate,Role,User,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,User
  +1 t_zV_r,rt,Role,User <= +2
 triangle1_Role,User,CourseSession,r: +1 t_zV_r,rt,Role,User +1 t_zV_r,lb,Role,CourseSession +1 t_zV_r,rt,User,CourseSession
  <= +2
 choose1_CertificateTemplate,Clinic: +1 t_zV_r,rt,CertificateTemplate,Clinic +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,CertificateTemplate,Clinic +1 t_zV_c,lb,CertificateTemplate,Clinic = +1
 triangle1_Client,User,Instructor,c: +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Client,Instructor +1 t_zV_c,rt,User,Instructor
  <= +2
 triangle1_Role,User,Instructor,c: +1 t_zV_c,rt,Role,User +1 t_zV_c,lb,Role,Instructor +1 t_zV_c,rt,User,Instructor
  <= +2
 triangle1_Client,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,Client,Therapist
  +1 t_zV_c,lb,Client,Instructor <= +2
 triangle1_Role,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,Role,Therapist
  +1 t_zV_c,lb,Role,Instructor <= +2
 triangle1_Clinic,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,lb,Clinic,Instructor
  +1 t_zV_c,rt,Clinic,Therapist <= +2
 triangle1_Location,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,lb,Location,Instructor
  +1 t_zV_c,rt,Location,Therapist <= +2
 triangle2_User,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,User,Clinic +1 t_zV_c,rt,User,Instructor
  <= +2
 triangle1_CertificateTemplate,Course,Instructor,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Instructor
  +1 t_zV_c,rt,Course,Instructor <= +2
 triangle2_Course,Role,Instructor,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,Instructor +1 t_zV_c,lb,Role,Instructor
  <= +2
 triangle2_Course,User,Instructor,c: +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Course,Instructor +1 t_zV_c,lb,User,Instructor
  <= +2
 triangle2_Course,Therapist,Instructor,c: +1 t_zV_c,lb,Course,Therapist +1 t_zV_c,rt,Course,Instructor
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 triangle1_Role,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,Role,Clinic +1 t_zV_r,lb,Role,Therapist
  <= +2
 axisVdifRT_c,Role,Clinic: -53 t_zV_c,rt,Role,Clinic -1 t_x_c,Clinic +1 t_x_c,Role >= -42
 triangle1_Client,Role,Clinic,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,Role,Clinic
  <= +2
 triangle1_Role,Clinic,Location,c: +1 t_zV_c,rt,Role,Clinic +1 t_zV_c,lb,Role,Location +1 t_zV_c,rt,Clinic,Location
  <= +2
 triangle1_Role,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,Role,Clinic +1 t_zV_c,lb,Role,Instructor
  <= +2
 axisVdifRT_r,Role,User: -49 t_zV_r,rt,Role,User -1 t_x_r,User +1 t_x_r,Role >= -42
 axisVdifRT_r,Role,Therapist: -52 t_zV_r,rt,Role,Therapist -1 t_x_r,Therapist +1 t_x_r,Role >= -42
 compDistAxis1_r,Role,User: +1 t_x_r,User -1 t_x_r,Role +1 t_dist_r,Role,User >= -0.5
 axisVdifRT_r,Role,Clinic: -50 t_zV_r,rt,Role,Clinic -1 t_x_r,Clinic +1 t_x_r,Role >= -42
 triangle1_CertificateTemplate,Role,Clinic,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,Role,Clinic <= +2
 triangle1_Role,Clinic,Location,r: +1 t_zV_r,rt,Role,Clinic +1 t_zV_r,lb,Role,Location +1 t_zV_r,rt,Clinic,Location
  <= +2
 triangle1_Client,Course,Instructor,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,Instructor +1 t_zV_c,rt,Course,Instructor
  <= +2
 choose1_Role,Clinic: +1 t_zV_r,rt,Role,Clinic +1 t_zV_c,rt,Role,Clinic +1 t_zV_r,lb,Role,Clinic +1 t_zV_c,lb,Role,Clinic
  = +1
 axisVdifRT_c,Course,Instructor: -54 t_zV_c,rt,Course,Instructor +1 t_x_c,Course -1 t_x_c,Instructor
  >= -42
 axisVdifRT_c,User,Instructor: -54 t_zV_c,rt,User,Instructor +1 t_x_c,User -1 t_x_c,Instructor >= -42
 axisVdifRT_c,Therapist,Instructor: -54 t_zV_c,rt,Therapist,Instructor +1 t_x_c,Therapist -1 t_x_c,Instructor
  >= -42
 axisVdifRT_r,CertificateTemplate,Instructor: -52 t_zV_r,rt,CertificateTemplate,Instructor -1 t_x_r,Instructor
  +1 t_x_r,CertificateTemplate >= -42
 triangle2_CertificateTemplate,Course,Instructor,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,Instructor
  +1 t_zV_r,lb,Course,Instructor <= +2
 axisVdifRT_c,CertificateTemplate,Instructor: -54 t_zV_c,rt,CertificateTemplate,Instructor +1 t_x_c,CertificateTemplate
  -1 t_x_c,Instructor >= -42
 triangle1_Client,CertificateTemplate,Instructor,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Instructor
  +1 t_zV_c,rt,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Role,Instructor,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,Instructor
  +1 t_zV_c,lb,Role,Instructor <= +2
 triangle2_CertificateTemplate,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Therapist,Instructor,c: +1 t_zV_c,lb,CertificateTemplate,Therapist +1 t_zV_c,rt,CertificateTemplate,Instructor
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 choose1_CertificateTemplate,Instructor: +1 t_zV_r,rt,CertificateTemplate,Instructor +1 t_zV_c,rt,CertificateTemplate,Instructor
  +1 t_zV_r,lb,CertificateTemplate,Instructor +1 t_zV_c,lb,CertificateTemplate,Instructor = +1
 axisVdifRT_r,Course,Location: -51 t_zV_r,rt,Course,Location -1 t_x_r,Location +1 t_x_r,Course >= -42
 triangle1_Course,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,Course,Location
  +1 t_zV_r,lb,Course,Instructor <= +2
 triangle2_Course,CourseSession,Location,r: +1 t_zV_r,lb,Course,CourseSession +1 t_zV_r,rt,Course,Location
  +1 t_zV_r,lb,CourseSession,Location <= +2
 triangle2_Course,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,Course,Clinic +1 t_zV_r,rt,Course,Location
  <= +2
 axisVdifLB_c,Course,Location: -52 t_zV_c,lb,Course,Location +1 t_x_c,Location -1 t_x_c,Course >= -42
 triangle2_Client,Course,Location,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,Location +1 t_zV_c,lb,Course,Location
  <= +2
 triangle2_Course,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,Course,Location
  +1 t_zV_c,rt,Course,Instructor <= +2
 triangle1_Client,Course,Location,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,Location +1 t_zV_r,rt,Course,Location
  <= +2
 choose1_Course,Location: +1 t_zV_r,rt,Course,Location +1 t_zV_c,rt,Course,Location +1 t_zV_r,lb,Course,Location
  +1 t_zV_c,lb,Course,Location = +1
 triangle2_User,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,User,Location
  +1 t_zV_c,rt,User,Instructor <= +2
 triangle2_Client,User,Location,c: +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Client,Location +1 t_zV_c,lb,User,Location
  <= +2
 axisVdifLB_c,User,Location: -53 t_zV_c,lb,User,Location +1 t_x_c,Location -1 t_x_c,User >= -42
 choose1_User,Location: +1 t_zV_r,rt,User,Location +1 t_zV_c,rt,User,Location +1 t_zV_r,lb,User,Location
  +1 t_zV_c,lb,User,Location = +1
 axisVdifRT_r,Course,Clinic: -50 t_zV_r,rt,Course,Clinic -1 t_x_r,Clinic +1 t_x_r,Course >= -42
 triangle1_CertificateTemplate,Course,Clinic,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,Course,Clinic <= +2
 triangle1_Course,Clinic,Location,r: +1 t_zV_r,rt,Course,Clinic +1 t_zV_r,lb,Course,Location +1 t_zV_r,rt,Clinic,Location
  <= +2
 triangle1_Course,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,Course,Clinic +1 t_zV_r,lb,Course,Therapist
  <= +2
 DistSumLB_Client,Instructor: +1 t_dist_r,Client,Instructor +1 t_dist_c,Client,Instructor >= +9.5
 triangle1_Course,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,Course,Clinic +1 t_zV_r,lb,Course,Instructor
  <= +2
 choose1_Course,Clinic: +1 t_zV_r,rt,Course,Clinic +1 t_zV_c,rt,Course,Clinic +1 t_zV_r,lb,Course,Clinic
  +1 t_zV_c,lb,Course,Clinic = +1
 triangle1_CertificateTemplate,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,CertificateTemplate,Instructor <= +2
 triangle1_Client,Role,Instructor,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,Instructor +1 t_zV_r,rt,Role,Instructor
  <= +2
 triangle1_Client,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,Client,Instructor
  <= +2
 triangle1_Client,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,Client,Location
  +1 t_zV_r,lb,Client,Instructor <= +2
 triangle1_Client,Clinic,Location,r: +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,Client,Location +1 t_zV_r,rt,Clinic,Location
  <= +2
 triangle1_Client,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,Client,Therapist
  <= +2
 triangle2_Client,Course,Clinic,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,Course,Clinic
  <= +2
 triangle2_Client,Course,Location,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,Location +1 t_zV_r,lb,Course,Location
  <= +2
 triangle2_Client,CourseSession,Location,r: +1 t_zV_r,lb,Client,CourseSession +1 t_zV_r,rt,Client,Location
  +1 t_zV_r,lb,CourseSession,Location <= +2
 triangle1_Client,CertificateTemplate,Instructor,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Instructor
  +1 t_zV_r,rt,CertificateTemplate,Instructor <= +2
 triangle1_Client,Course,Instructor,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,Instructor +1 t_zV_r,rt,Course,Instructor
  <= +2
 triangle1_CertificateTemplate,Course,Instructor,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,Instructor
  +1 t_zV_r,rt,Course,Instructor <= +2
 triangle1_CertificateTemplate,Role,Instructor,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,Instructor
  +1 t_zV_r,rt,Role,Instructor <= +2
 DistAxisLB1_c,Client,Instructor: +11.5 t_zV_r,rt,Client,Instructor +11.5 t_zV_r,lb,Client,Instructor
  +1 t_dist_c,Client,Instructor >= +11.5
 triangle2_Client,Course,Instructor,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,Instructor +1 t_zV_r,lb,Course,Instructor
  <= +2
 triangle2_Client,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,Client,Location
  +1 t_zV_r,rt,Client,Instructor <= +2
 triangle2_Course,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,Course,Location
  +1 t_zV_r,rt,Course,Instructor <= +2
 triangle2_Role,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,Role,Location
  +1 t_zV_r,rt,Role,Instructor <= +2
 triangle2_CourseSession,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,CourseSession,Location
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle2_Clinic,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,rt,Clinic,Instructor
  +1 t_zV_r,lb,Clinic,Location <= +2
 triangle1_Clinic,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,lb,Clinic,Instructor
  +1 t_zV_r,rt,Clinic,Location <= +2
 triangle1_Clinic,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,lb,Clinic,Instructor
  +1 t_zV_r,rt,Clinic,Therapist <= +2
 triangle2_CertificateTemplate,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,CertificateTemplate,Instructor <= +2
 triangle2_Course,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,Course,Clinic +1 t_zV_r,rt,Course,Instructor
  <= +2
 triangle2_Role,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,Role,Clinic +1 t_zV_r,rt,Role,Instructor
  <= +2
 triangle1_CertificateTemplate,CourseSession,Instructor,r: +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,Instructor +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle2_Client,CertificateTemplate,Instructor,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Instructor
  +1 t_zV_r,lb,CertificateTemplate,Instructor <= +2
 triangle2_Client,CertificateTemplate,Clinic,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Clinic
  +1 t_zV_r,lb,CertificateTemplate,Clinic <= +2
 triangle2_Client,CertificateTemplate,Role,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Role
  +1 t_zV_r,lb,CertificateTemplate,Role <= +2
 triangle2_Client,CertificateTemplate,Course,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Course
  +1 t_zV_r,lb,CertificateTemplate,Course <= +2
 axisVdifLB_r,Client,CertificateTemplate: -51 t_zV_r,lb,Client,CertificateTemplate +1 t_x_r,CertificateTemplate
  -1 t_x_r,Client >= -42
 axisVdifLB_r,Client,Therapist: -51 t_zV_r,lb,Client,Therapist +1 t_x_r,Therapist -1 t_x_r,Client
  >= -42
 axisVdifRT_r,Client,Clinic: -50 t_zV_r,rt,Client,Clinic -1 t_x_r,Clinic +1 t_x_r,Client >= -42
 axisVdifRT_r,Client,Location: -51 t_zV_r,rt,Client,Location -1 t_x_r,Location +1 t_x_r,Client >= -42
 axisVdifLB_r,Client,Instructor: -51 t_zV_r,lb,Client,Instructor +1 t_x_r,Instructor -1 t_x_r,Client
  >= -42
 axisVdifLB_r,CertificateTemplate,Instructor: -49 t_zV_r,lb,CertificateTemplate,Instructor +1 t_x_r,Instructor
  -1 t_x_r,CertificateTemplate >= -42
 axisVdifLB_r,Clinic,Instructor: -50 t_zV_r,lb,Clinic,Instructor +1 t_x_r,Instructor -1 t_x_r,Clinic
  >= -42
 axisVdifLB_r,Location,Instructor: -51 t_zV_r,lb,Location,Instructor +1 t_x_r,Instructor -1 t_x_r,Location
  >= -42
 axisVdifRT_r,Client,Instructor: -52 t_zV_r,rt,Client,Instructor -1 t_x_r,Instructor +1 t_x_r,Client
  >= -42
 axisVdifRT_r,Therapist,Instructor: -52 t_zV_r,rt,Therapist,Instructor -1 t_x_r,Instructor +1 t_x_r,Therapist
  >= -42
 axisVdifLB_r,Course,Therapist: -48 t_zV_r,lb,Course,Therapist +1 t_x_r,Therapist -1 t_x_r,Course
  >= -42
 triangle1_Role,CourseSession,Location,r: +1 t_zV_r,rt,Role,CourseSession +1 t_zV_r,lb,Role,Location
  +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle2_Client,Course,User,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Course,User
  <= +2
 triangle1_Client,Course,User,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Course,User
  <= +2
 triangle1_CertificateTemplate,Course,User,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,User
  +1 t_zV_r,rt,Course,User <= +2
 triangle1_Course,User,CourseSession,r: +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Course,CourseSession
  +1 t_zV_r,rt,User,CourseSession <= +2
 triangle1_Course,User,Clinic,r: +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Course,Clinic +1 t_zV_r,rt,User,Clinic
  <= +2
 triangle1_Course,User,Location,r: +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Course,Location +1 t_zV_r,rt,User,Location
  <= +2
 triangle1_Course,User,Therapist,r: +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Course,Therapist +1 t_zV_r,rt,User,Therapist
  <= +2
 triangle1_Course,User,Instructor,r: +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Course,Instructor +1 t_zV_r,rt,User,Instructor
  <= +2
 triangle2_Role,User,Clinic,r: +1 t_zV_r,lb,Role,User +1 t_zV_r,rt,Role,Clinic +1 t_zV_r,lb,User,Clinic
  <= +2
 triangle2_User,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,User,Clinic +1 t_zV_r,rt,User,Location
  <= +2
 triangle2_User,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,User,Clinic +1 t_zV_r,rt,User,Therapist
  <= +2
 centerDistAxis2_r,CourseSession: -1 t_x_r,CourseSession +1 t_center_dist_r,CourseSession >= -17.5
 axisVdifRT_r,CourseSession,Location: -51 t_zV_r,rt,CourseSession,Location -1 t_x_r,Location +1 t_x_r,CourseSession
  >= -42
 triangle2_User,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,User,Clinic +1 t_zV_r,rt,User,Instructor
  <= +2
 triangle2_Client,User,Clinic,r: +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,User,Clinic
  <= +2
 DistSumLB_User,Instructor: +1 t_dist_r,User,Instructor +1 t_dist_c,User,Instructor >= +8.5
 DistAxisLB1_r,User,Instructor: +8.5 t_zV_c,rt,User,Instructor +8.5 t_zV_c,lb,User,Instructor +1 t_dist_r,User,Instructor
  >= +8.5
 triangle2_CertificateTemplate,User,Clinic,r: +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_r,rt,CertificateTemplate,Clinic
  +1 t_zV_r,lb,User,Clinic <= +2
 triangle2_Course,User,Clinic,r: +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Course,Clinic +1 t_zV_r,lb,User,Clinic
  <= +2
 DistSumLB_Client,User: +1 t_dist_r,Client,User +1 t_dist_c,Client,User >= +8
 DistAxisLB1_r,Client,User: +8 t_zV_c,rt,Client,User +8 t_zV_c,lb,Client,User +1 t_dist_r,Client,User
  >= +8
 triangle1_Client,User,CourseSession,r: +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Client,CourseSession
  +1 t_zV_r,rt,User,CourseSession <= +2
 triangle1_Client,User,Location,r: +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Client,Location +1 t_zV_r,rt,User,Location
  <= +2
 triangle1_Client,User,Therapist,r: +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Client,Therapist +1 t_zV_r,rt,User,Therapist
  <= +2
 triangle1_Client,User,Instructor,r: +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Client,Instructor +1 t_zV_r,rt,User,Instructor
  <= +2
 triangle2_Client,CertificateTemplate,User,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,User
  +1 t_zV_r,lb,CertificateTemplate,User <= +2
 triangle2_User,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,User,Location
  +1 t_zV_r,rt,User,Instructor <= +2
 triangle2_User,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,User,Location +1 t_zV_r,rt,User,Therapist
  <= +2
 triangle2_Role,User,Location,r: +1 t_zV_r,lb,Role,User +1 t_zV_r,rt,Role,Location +1 t_zV_r,lb,User,Location
  <= +2
 triangle2_Course,User,Location,r: +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Course,Location +1 t_zV_r,lb,User,Location
  <= +2
 triangle2_Client,User,Location,r: +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Client,Location +1 t_zV_r,lb,User,Location
  <= +2
 triangle1_User,Clinic,Location,r: +1 t_zV_r,rt,User,Clinic +1 t_zV_r,lb,User,Location +1 t_zV_r,rt,Clinic,Location
  <= +2
 triangle1_User,CourseSession,Location,r: +1 t_zV_r,rt,User,CourseSession +1 t_zV_r,lb,User,Location
  +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle2_CourseSession,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,CourseSession,Clinic
  +1 t_zV_c,rt,CourseSession,Location <= +2
 triangle2_Role,CourseSession,Location,c: +1 t_zV_c,lb,Role,CourseSession +1 t_zV_c,rt,Role,Location
  +1 t_zV_c,lb,CourseSession,Location <= +2
 triangle1_Client,CertificateTemplate,Clinic,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Clinic
  +1 t_zV_c,rt,CertificateTemplate,Clinic <= +2
 triangle1_CertificateTemplate,Clinic,Location,c: +1 t_zV_c,rt,CertificateTemplate,Clinic +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,Clinic,Location <= +2
 triangle1_CertificateTemplate,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_c,lb,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_c,lb,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,CourseSession,Clinic,c: +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,CertificateTemplate,Clinic +1 t_zV_c,lb,CourseSession,Clinic <= +2
 axisVdifLB_r,User,CourseSession: -49 t_zV_r,lb,User,CourseSession +1 t_x_r,CourseSession -1 t_x_r,User
  >= -42
 axisVdifLB_r,User,Clinic: -49 t_zV_r,lb,User,Clinic +1 t_x_r,Clinic -1 t_x_r,User >= -42
 axisVdifLB_r,User,Location: -49 t_zV_r,lb,User,Location +1 t_x_r,Location -1 t_x_r,User >= -42
 axisVdifRT_r,User,CourseSession: -51 t_zV_r,rt,User,CourseSession -1 t_x_r,CourseSession +1 t_x_r,User
  >= -42
 compDistAxis2_r,User,Therapist: -1 t_x_r,Therapist +1 t_x_r,User +1 t_dist_r,User,Therapist >= +1.5
 triangle2_CertificateTemplate,User,CourseSession,r: +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,User,CourseSession <= +2
 triangle1_Clinic,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,lb,Clinic,Therapist
  +1 t_zV_c,rt,Clinic,Location <= +2
 centerDistAxis2_r,Course: -1 t_x_r,Course +1 t_center_dist_r,Course >= -19
 triangle2_Client,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,Client,Therapist
  <= +2
 triangle2_Clinic,Therapist,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,lb,Clinic,Therapist
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 axisVdifLB_r,Course,User: -48 t_zV_r,lb,Course,User +1 t_x_r,User -1 t_x_r,Course >= -42
 triangle2_Role,User,CourseSession,r: +1 t_zV_r,lb,Role,User +1 t_zV_r,rt,Role,CourseSession +1 t_zV_r,lb,User,CourseSession
  <= +2
 triangle2_User,CourseSession,Location,r: +1 t_zV_r,lb,User,CourseSession +1 t_zV_r,rt,User,Location
  +1 t_zV_r,lb,CourseSession,Location <= +2
 axisVdifLB_r,User,Therapist: -49 t_zV_r,lb,User,Therapist +1 t_x_r,Therapist -1 t_x_r,User >= -42
 triangle1_User,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,User,Clinic +1 t_zV_r,lb,User,Therapist
  <= +2
 triangle2_Role,User,Therapist,r: +1 t_zV_r,lb,Role,User +1 t_zV_r,rt,Role,Therapist +1 t_zV_r,lb,User,Therapist
  <= +2
 axisVdifRT_r,Course,User: -49 t_zV_r,rt,Course,User -1 t_x_r,User +1 t_x_r,Course >= -42
 centerDistAxis1_r,User: +1 t_x_r,User +1 t_center_dist_r,User >= +18.5
 centerDistAxis2_r,Instructor: -1 t_x_r,Instructor +1 t_center_dist_r,Instructor >= -17
 triangle1_Client,Course,Clinic,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,Course,Clinic
  <= +2
 triangle1_User,CourseSession,Instructor,r: +1 t_zV_r,rt,User,CourseSession +1 t_zV_r,lb,User,Instructor
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle1_User,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,User,Clinic +1 t_zV_r,lb,User,Instructor
  <= +2
 triangle1_User,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,User,Location
  +1 t_zV_r,lb,User,Instructor <= +2
 centerDistAxis2_r,Client: -1 t_x_r,Client +1 t_center_dist_r,Client >= -17.5
 triangle2_Client,User,Instructor,r: +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Client,Instructor +1 t_zV_r,lb,User,Instructor
  <= +2
 triangle2_CertificateTemplate,User,Instructor,r: +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_r,rt,CertificateTemplate,Instructor
  +1 t_zV_r,lb,User,Instructor <= +2
 triangle2_Course,User,Instructor,r: +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Course,Instructor +1 t_zV_r,lb,User,Instructor
  <= +2
 triangle2_Role,User,Instructor,r: +1 t_zV_r,lb,Role,User +1 t_zV_r,rt,Role,Instructor +1 t_zV_r,lb,User,Instructor
  <= +2
 triangle1_Client,User,Clinic,c: +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Client,Clinic +1 t_zV_c,rt,User,Clinic
  <= +2
 triangle1_User,Clinic,Location,c: +1 t_zV_c,rt,User,Clinic +1 t_zV_c,lb,User,Location +1 t_zV_c,rt,Clinic,Location
  <= +2
 triangle1_User,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,User,Clinic +1 t_zV_c,lb,User,Instructor
  <= +2
 triangle1_CertificateTemplate,Course,Clinic,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,Course,Clinic <= +2
 triangle1_Course,Clinic,Location,c: +1 t_zV_c,rt,Course,Clinic +1 t_zV_c,lb,Course,Location +1 t_zV_c,rt,Clinic,Location
  <= +2
 triangle1_Course,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,Course,Clinic +1 t_zV_c,lb,Course,Therapist
  <= +2
 axisVdifRT_r,User,Instructor: -52 t_zV_r,rt,User,Instructor -1 t_x_r,Instructor +1 t_x_r,User >= -42
 triangle1_Course,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,Course,Clinic +1 t_zV_c,lb,Course,Instructor
  <= +2
 triangle2_Course,User,Clinic,c: +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Course,Clinic +1 t_zV_c,lb,User,Clinic
  <= +2
 compDistAxis2_r,User,Instructor: -1 t_x_r,Instructor +1 t_x_r,User +1 t_dist_r,User,Instructor >= +1.5
 axisVdifLB_r,Client,User: -51 t_zV_r,lb,Client,User +1 t_x_r,User -1 t_x_r,Client >= -42
 triangle2_User,CourseSession,Clinic,c: +1 t_zV_c,lb,User,CourseSession +1 t_zV_c,rt,User,Clinic +1 t_zV_c,lb,CourseSession,Clinic
  <= +2
 triangle1_Role,User,CourseSession,c: +1 t_zV_c,rt,Role,User +1 t_zV_c,lb,Role,CourseSession +1 t_zV_c,rt,User,CourseSession
  <= +2
 triangle2_CertificateTemplate,Role,CourseSession,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,Role,CourseSession <= +2
 triangle2_Role,CourseSession,Clinic,c: +1 t_zV_c,lb,Role,CourseSession +1 t_zV_c,rt,Role,Clinic +1 t_zV_c,lb,CourseSession,Clinic
  <= +2
 axisVdifRT_r,Client,User: -49 t_zV_r,rt,Client,User -1 t_x_r,User +1 t_x_r,Client >= -42
 triangle1_CertificateTemplate,CourseSession,Location,c: +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,CertificateTemplate,Location +1 t_zV_c,rt,CourseSession,Location <= +2
 triangle1_Role,CourseSession,Location,c: +1 t_zV_c,rt,Role,CourseSession +1 t_zV_c,lb,Role,Location
  +1 t_zV_c,rt,CourseSession,Location <= +2
 compDistAxis1_r,Client,User: +1 t_x_r,User -1 t_x_r,Client +1 t_dist_r,Client,User >= +1
 triangle1_User,CourseSession,Location,c: +1 t_zV_c,rt,User,CourseSession +1 t_zV_c,lb,User,Location
  +1 t_zV_c,rt,CourseSession,Location <= +2
 triangle2_User,CourseSession,Therapist,r: +1 t_zV_r,lb,User,CourseSession +1 t_zV_r,rt,User,Therapist
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 triangle1_User,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,User,Therapist
  +1 t_zV_r,lb,User,Instructor <= +2
 DistAxisLB1_c,User,Therapist: +11 t_zV_r,rt,User,Therapist +11 t_zV_r,lb,User,Therapist +1 t_dist_c,User,Therapist
  >= +11
 choose1_User,Therapist: +1 t_zV_r,rt,User,Therapist +1 t_zV_c,rt,User,Therapist +1 t_zV_r,lb,User,Therapist
  +1 t_zV_c,lb,User,Therapist = +1
 axisVdifRT_c,CertificateTemplate,Clinic: -53 t_zV_c,rt,CertificateTemplate,Clinic -1 t_x_c,Clinic
  +1 t_x_c,CertificateTemplate >= -42
 axisVdifRT_c,Course,Clinic: -53 t_zV_c,rt,Course,Clinic -1 t_x_c,Clinic +1 t_x_c,Course >= -42
 axisVdifRT_c,User,Clinic: -53 t_zV_c,rt,User,Clinic -1 t_x_c,Clinic +1 t_x_c,User >= -42
 axisVdifLB_c,Role,CourseSession: -52 t_zV_c,lb,Role,CourseSession +1 t_x_c,CourseSession -1 t_x_c,Role
  >= -42
 axisVdifLB_c,User,CourseSession: -53 t_zV_c,lb,User,CourseSession +1 t_x_c,CourseSession -1 t_x_c,User
  >= -42
 axisVdifRT_c,CourseSession,Location: -53 t_zV_c,rt,CourseSession,Location -1 t_x_c,Location +1 t_x_c,CourseSession
  >= -42
 triangle1_CertificateTemplate,CourseSession,Clinic,r: +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,Clinic +1 t_zV_r,rt,CourseSession,Clinic <= +2
 triangle1_Role,CourseSession,Clinic,r: +1 t_zV_r,rt,Role,CourseSession +1 t_zV_r,lb,Role,Clinic +1 t_zV_r,rt,CourseSession,Clinic
  <= +2
 triangle1_User,CourseSession,Clinic,r: +1 t_zV_r,rt,User,CourseSession +1 t_zV_r,lb,User,Clinic +1 t_zV_r,rt,CourseSession,Clinic
  <= +2
 triangle1_CourseSession,Clinic,Location,r: +1 t_zV_r,rt,CourseSession,Clinic +1 t_zV_r,lb,CourseSession,Location
  +1 t_zV_r,rt,Clinic,Location <= +2
 triangle1_CourseSession,Clinic,Therapist,r: +1 t_zV_r,rt,Clinic,Therapist +1 t_zV_r,rt,CourseSession,Clinic
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 axisVdifRT_c,CourseSession,Clinic: -53 t_zV_c,rt,CourseSession,Clinic -1 t_x_c,Clinic +1 t_x_c,CourseSession
  >= -42
 triangle1_CertificateTemplate,CourseSession,Clinic,c: +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,CertificateTemplate,Clinic +1 t_zV_c,rt,CourseSession,Clinic <= +2
 triangle1_User,CourseSession,Clinic,c: +1 t_zV_c,rt,User,CourseSession +1 t_zV_c,lb,User,Clinic +1 t_zV_c,rt,CourseSession,Clinic
  <= +2
 triangle1_CourseSession,Clinic,Location,c: +1 t_zV_c,rt,CourseSession,Clinic +1 t_zV_c,lb,CourseSession,Location
  +1 t_zV_c,rt,Clinic,Location <= +2
 triangle2_Client,CourseSession,Clinic,r: +1 t_zV_r,lb,Client,CourseSession +1 t_zV_r,rt,Client,Clinic
  +1 t_zV_r,lb,CourseSession,Clinic <= +2
 triangle2_CertificateTemplate,CourseSession,Clinic,r: +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,CertificateTemplate,Clinic +1 t_zV_r,lb,CourseSession,Clinic <= +2
 triangle2_Course,CourseSession,Clinic,r: +1 t_zV_r,lb,Course,CourseSession +1 t_zV_r,rt,Course,Clinic
  +1 t_zV_r,lb,CourseSession,Clinic <= +2
 DistSumLB_Therapist,CourseSession: +1 t_dist_r,Therapist,CourseSession +1 t_dist_c,Therapist,CourseSession
  >= +9.5
 triangle2_Role,CourseSession,Clinic,r: +1 t_zV_r,lb,Role,CourseSession +1 t_zV_r,rt,Role,Clinic +1 t_zV_r,lb,CourseSession,Clinic
  <= +2
 triangle2_User,CourseSession,Clinic,r: +1 t_zV_r,lb,User,CourseSession +1 t_zV_r,rt,User,Clinic +1 t_zV_r,lb,CourseSession,Clinic
  <= +2
 triangle2_CourseSession,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,CourseSession,Clinic
  +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle2_CourseSession,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,CourseSession,Clinic
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 axisVdifLB_c,Clinic,Therapist: -53 t_zV_c,lb,Clinic,Therapist +1 t_x_c,Therapist -1 t_x_c,Clinic
  >= -42
 axisVdifLB_r,CourseSession,Clinic: -51 t_zV_r,lb,CourseSession,Clinic +1 t_x_r,Clinic -1 t_x_r,CourseSession
  >= -42
 axisVdifRT_r,CourseSession,Clinic: -50 t_zV_r,rt,CourseSession,Clinic -1 t_x_r,Clinic +1 t_x_r,CourseSession
  >= -42
 compDistAxis1_r,Course,CourseSession: +1 t_x_r,CourseSession -1 t_x_r,Course +1 t_dist_r,Course,CourseSession
  >= -1.5
 compDistAxis2_r,CourseSession,Instructor: -1 t_x_r,Instructor +1 t_x_r,CourseSession +1 t_dist_r,CourseSession,Instructor
  >= +0.5
 compDistAxis2_r,Therapist,CourseSession: +1 t_x_r,Therapist -1 t_x_r,CourseSession +1 t_dist_r,Therapist,CourseSession
  >= -0.5
 axisVdifLB_r,User,Instructor: -49 t_zV_r,lb,User,Instructor +1 t_x_r,Instructor -1 t_x_r,User >= -42
 triangle1_Client,CertificateTemplate,Therapist,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Therapist
  +1 t_zV_r,rt,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,CertificateTemplate,Therapist
  +1 t_zV_r,lb,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Course,Therapist,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,Therapist
  +1 t_zV_r,lb,Course,Therapist <= +2
 triangle2_CertificateTemplate,Role,Therapist,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,Therapist
  +1 t_zV_r,lb,Role,Therapist <= +2
 triangle2_CertificateTemplate,User,Therapist,r: +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_r,rt,CertificateTemplate,Therapist
  +1 t_zV_r,lb,User,Therapist <= +2
 triangle2_CertificateTemplate,CourseSession,Therapist,r: +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,CertificateTemplate,Therapist +1 t_zV_r,lb,CourseSession,Therapist <= +2
 triangle2_CertificateTemplate,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,CertificateTemplate,Therapist <= +2
 choose1_CourseSession,Clinic: +1 t_zV_r,rt,CourseSession,Clinic +1 t_zV_c,rt,CourseSession,Clinic
  +1 t_zV_r,lb,CourseSession,Clinic +1 t_zV_c,lb,CourseSession,Clinic = +1
 triangle1_Client,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,Client,Therapist
  +1 t_zV_r,lb,Client,Instructor <= +2
 triangle2_Client,CertificateTemplate,Therapist,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Therapist
  +1 t_zV_r,lb,CertificateTemplate,Therapist <= +2
 axisVdifRT_r,CourseSession,Therapist: -52 t_zV_r,rt,CourseSession,Therapist -1 t_x_r,Therapist +1 t_x_r,CourseSession
  >= -42
 triangle1_CertificateTemplate,CourseSession,Therapist,r: +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,Therapist +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle1_Role,CourseSession,Therapist,r: +1 t_zV_r,rt,Role,CourseSession +1 t_zV_r,lb,Role,Therapist
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle1_User,CourseSession,Therapist,r: +1 t_zV_r,rt,User,CourseSession +1 t_zV_r,lb,User,Therapist
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle2_CourseSession,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,CourseSession,Clinic
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle2_CourseSession,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,CourseSession,Location
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 axisVdifRT_c,CourseSession,Therapist: -53 t_zV_c,rt,CourseSession,Therapist -1 t_x_c,Therapist +1 t_x_c,CourseSession
  >= -42
 DistAxisLB1_r,Therapist,CourseSession: +9.5 t_zV_c,rt,CourseSession,Therapist +9.5 t_zV_c,lb,CourseSession,Therapist
  +1 t_dist_r,Therapist,CourseSession >= +9.5
 triangle1_CertificateTemplate,CourseSession,Therapist,c: +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,CertificateTemplate,Therapist +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle2_CourseSession,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,CourseSession,Clinic
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 axisVdifLB_c,CourseSession,Therapist: -56 t_zV_c,lb,CourseSession,Therapist +1 t_x_c,Therapist -1 t_x_c,CourseSession
  >= -42
 triangle1_CourseSession,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,CourseSession,Clinic
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 triangle1_CourseSession,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,CourseSession,Location
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 triangle2_Role,CourseSession,Therapist,c: +1 t_zV_c,lb,Role,CourseSession +1 t_zV_c,rt,Role,Therapist
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 axisVdifRT_r,Client,Therapist: -52 t_zV_r,rt,Client,Therapist -1 t_x_r,Therapist +1 t_x_r,Client
  >= -42
 compDistAxis1_r,Client,CourseSession: +1 t_x_r,CourseSession -1 t_x_r,Client +1 t_dist_r,Client,CourseSession
  >= +0
 DistSumLB_Client,CourseSession: +1 t_dist_r,Client,CourseSession +1 t_dist_c,Client,CourseSession
  >= +9
 centerDistAxis1_r,Therapist: +1 t_x_r,Therapist +1 t_center_dist_r,Therapist >= +17
 axisVdifLB_r,Therapist,Instructor: -52 t_zV_r,lb,Therapist,Instructor +1 t_x_r,Instructor -1 t_x_r,Therapist
  >= -42
 axisVdifRT_r,CertificateTemplate,Therapist: -52 t_zV_r,rt,CertificateTemplate,Therapist -1 t_x_r,Therapist
  +1 t_x_r,CertificateTemplate >= -42
 axisVdifRT_r,Course,Therapist: -52 t_zV_r,rt,Course,Therapist -1 t_x_r,Therapist +1 t_x_r,Course
  >= -42
 triangle2_Client,Course,Therapist,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,Therapist +1 t_zV_r,lb,Course,Therapist
  <= +2
 triangle2_Client,User,Therapist,r: +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Client,Therapist +1 t_zV_r,lb,User,Therapist
  <= +2
 triangle2_Client,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,Client,Location
  +1 t_zV_r,rt,Client,Therapist <= +2
 triangle1_Course,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,Course,Therapist
  +1 t_zV_r,lb,Course,Instructor <= +2
 triangle2_Course,User,Therapist,r: +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Course,Therapist +1 t_zV_r,lb,User,Therapist
  <= +2
 triangle2_Course,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,Course,Clinic +1 t_zV_r,rt,Course,Therapist
  <= +2
 triangle2_Course,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,Course,Location
  +1 t_zV_r,rt,Course,Therapist <= +2
 triangle2_Course,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,Course,Therapist
  +1 t_zV_r,rt,Course,Instructor <= +2
 triangle2_Role,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,Role,Therapist
  +1 t_zV_r,rt,Role,Instructor <= +2
 triangle2_User,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,User,Therapist
  +1 t_zV_r,rt,User,Instructor <= +2
 triangle2_Clinic,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,rt,Clinic,Instructor
  +1 t_zV_r,lb,Clinic,Therapist <= +2
 triangle2_Location,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,rt,Location,Instructor
  +1 t_zV_r,lb,Location,Therapist <= +2
 triangle2_Client,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,Client,Therapist
  +1 t_zV_r,rt,Client,Instructor <= +2
 triangle2_CertificateTemplate,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,CertificateTemplate,Therapist
  +1 t_zV_r,rt,CertificateTemplate,Instructor <= +2
 triangle1_Client,Course,Therapist,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,Therapist +1 t_zV_r,rt,Course,Therapist
  <= +2
 triangle1_CertificateTemplate,Course,Therapist,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,Therapist
  +1 t_zV_r,rt,Course,Therapist <= +2
 triangle2_CourseSession,Therapist,Instructor,r: +1 t_zV_r,lb,Therapist,Instructor +1 t_zV_r,lb,CourseSession,Therapist
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle2_Course,CourseSession,Therapist,r: +1 t_zV_r,lb,Course,CourseSession +1 t_zV_r,rt,Course,Therapist
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 triangle2_Client,CourseSession,Therapist,r: +1 t_zV_r,lb,Client,CourseSession +1 t_zV_r,rt,Client,Therapist
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 DistAxisLB1_c,Therapist,CourseSession: +12.5 t_zV_r,rt,CourseSession,Therapist +12.5 t_zV_r,lb,CourseSession,Therapist
  +1 t_dist_c,Therapist,CourseSession >= +12.5
 choose1_CourseSession,Therapist: +1 t_zV_r,rt,CourseSession,Therapist +1 t_zV_c,rt,CourseSession,Therapist
  +1 t_zV_r,lb,CourseSession,Therapist +1 t_zV_c,lb,CourseSession,Therapist = +1
 triangle1_CertificateTemplate,Course,Location,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,Course,Location <= +2
 triangle1_Course,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,Course,Location
  +1 t_zV_c,lb,Course,Therapist <= +2
 triangle1_Course,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,Course,Location
  +1 t_zV_c,lb,Course,Instructor <= +2
 triangle2_Course,Role,Location,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,Location +1 t_zV_c,lb,Role,Location
  <= +2
 triangle2_Course,User,Location,c: +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Course,Location +1 t_zV_c,lb,User,Location
  <= +2
 triangle1_Course,User,Instructor,c: +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Course,Instructor +1 t_zV_c,rt,User,Instructor
  <= +2
 triangle2_Course,Role,User,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Role,User
  <= +2
 axisVdifRT_r,Client,CourseSession: -51 t_zV_r,rt,Client,CourseSession -1 t_x_r,CourseSession +1 t_x_r,Client
  >= -42
 triangle1_Client,CourseSession,Location,r: +1 t_zV_r,rt,Client,CourseSession +1 t_zV_r,lb,Client,Location
  +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle1_Client,CourseSession,Therapist,r: +1 t_zV_r,rt,Client,CourseSession +1 t_zV_r,lb,Client,Therapist
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle1_Client,CourseSession,Instructor,r: +1 t_zV_r,rt,Client,CourseSession +1 t_zV_r,lb,Client,Instructor
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle2_Client,CertificateTemplate,CourseSession,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,CourseSession <= +2
 triangle2_Client,Course,CourseSession,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,CourseSession
  +1 t_zV_r,lb,Course,CourseSession <= +2
 triangle2_Client,User,CourseSession,r: +1 t_zV_r,lb,Client,User +1 t_zV_r,rt,Client,CourseSession
  +1 t_zV_r,lb,User,CourseSession <= +2
 axisVdifRT_c,Client,CourseSession: -56 t_zV_c,rt,Client,CourseSession -1 t_x_c,CourseSession +1 t_x_c,Client
  >= -42
 DistAxisLB1_r,Client,CourseSession: +9 t_zV_c,rt,Client,CourseSession +9 t_zV_c,lb,Client,CourseSession
  +1 t_dist_r,Client,CourseSession >= +9
 triangle1_Client,CourseSession,Clinic,c: +1 t_zV_c,rt,Client,CourseSession +1 t_zV_c,lb,Client,Clinic
  +1 t_zV_c,rt,CourseSession,Clinic <= +2
 triangle1_Client,CourseSession,Location,c: +1 t_zV_c,rt,Client,CourseSession +1 t_zV_c,lb,Client,Location
  +1 t_zV_c,rt,CourseSession,Location <= +2
 triangle1_Client,CourseSession,Therapist,c: +1 t_zV_c,rt,Client,CourseSession +1 t_zV_c,lb,Client,Therapist
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle2_Client,CertificateTemplate,CourseSession,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,CourseSession
  +1 t_zV_c,lb,CertificateTemplate,CourseSession <= +2
 triangle2_Client,User,CourseSession,c: +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Client,CourseSession
  +1 t_zV_c,lb,User,CourseSession <= +2
 axisVdifLB_c,Client,CourseSession: -53 t_zV_c,lb,Client,CourseSession +1 t_x_c,CourseSession -1 t_x_c,Client
  >= -42
 triangle1_Client,CertificateTemplate,CourseSession,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,CourseSession
  +1 t_zV_c,rt,CertificateTemplate,CourseSession <= +2
 triangle1_Client,Role,CourseSession,c: +1 t_zV_c,rt,Client,Role +1 t_zV_c,lb,Client,CourseSession
  +1 t_zV_c,rt,Role,CourseSession <= +2
 triangle1_Client,User,CourseSession,c: +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Client,CourseSession
  +1 t_zV_c,rt,User,CourseSession <= +2
 triangle2_Client,CourseSession,Clinic,c: +1 t_zV_c,lb,Client,CourseSession +1 t_zV_c,rt,Client,Clinic
  +1 t_zV_c,lb,CourseSession,Clinic <= +2
 triangle2_Client,CourseSession,Location,c: +1 t_zV_c,lb,Client,CourseSession +1 t_zV_c,rt,Client,Location
  +1 t_zV_c,lb,CourseSession,Location <= +2
 triangle2_Client,CourseSession,Therapist,c: +1 t_zV_c,lb,Client,CourseSession +1 t_zV_c,rt,Client,Therapist
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 triangle2_Role,User,CourseSession,c: +1 t_zV_c,lb,Role,User +1 t_zV_c,rt,Role,CourseSession +1 t_zV_c,lb,User,CourseSession
  <= +2
 triangle1_Role,CourseSession,Therapist,c: +1 t_zV_c,rt,Role,CourseSession +1 t_zV_c,lb,Role,Therapist
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle1_Role,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,Role,Clinic +1 t_zV_c,lb,Role,Therapist
  <= +2
 triangle1_CertificateTemplate,Role,CourseSession,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,Role,CourseSession <= +2
 triangle1_CertificateTemplate,Role,Clinic,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,Role,Clinic <= +2
 triangle1_CertificateTemplate,Role,Location,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,Role,Location <= +2
 triangle1_CertificateTemplate,Role,Therapist,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,Therapist
  +1 t_zV_c,rt,Role,Therapist <= +2
 triangle1_CertificateTemplate,Role,Instructor,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,Instructor
  +1 t_zV_c,rt,Role,Instructor <= +2
 triangle1_Role,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,Role,Location +1 t_zV_c,lb,Role,Therapist
  <= +2
 triangle2_Role,Therapist,Instructor,c: +1 t_zV_c,lb,Role,Therapist +1 t_zV_c,rt,Role,Instructor +1 t_zV_c,lb,Therapist,Instructor
  <= +2
 triangle2_Role,User,Clinic,c: +1 t_zV_c,lb,Role,User +1 t_zV_c,rt,Role,Clinic +1 t_zV_c,lb,User,Clinic
  <= +2
 triangle2_Role,User,Location,c: +1 t_zV_c,lb,Role,User +1 t_zV_c,rt,Role,Location +1 t_zV_c,lb,User,Location
  <= +2
 triangle2_Role,User,Instructor,c: +1 t_zV_c,lb,Role,User +1 t_zV_c,rt,Role,Instructor +1 t_zV_c,lb,User,Instructor
  <= +2
 triangle1_Client,Course,User,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Course,User
  <= +2
 triangle1_Client,Course,Therapist,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,Therapist +1 t_zV_c,rt,Course,Therapist
  <= +2
 triangle1_CertificateTemplate,Course,Therapist,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Therapist
  +1 t_zV_c,rt,Course,Therapist <= +2
 triangle1_Course,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,Course,Therapist
  +1 t_zV_c,lb,Course,Instructor <= +2
 triangle2_Course,Role,Therapist,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,Therapist +1 t_zV_c,lb,Role,Therapist
  <= +2
 triangle1_Client,Course,Location,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,Location +1 t_zV_c,rt,Course,Location
  <= +2
 DistAxisLB1_c,Client,CourseSession: +12.5 t_zV_r,rt,Client,CourseSession +12.5 t_zV_r,lb,Client,CourseSession
  +1 t_dist_c,Client,CourseSession >= +12.5
 choose1_Client,CourseSession: +1 t_zV_r,rt,Client,CourseSession +1 t_zV_c,rt,Client,CourseSession
  +1 t_zV_r,lb,Client,CourseSession +1 t_zV_c,lb,Client,CourseSession = +1
 triangle1_CertificateTemplate,Course,Role,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,Role
  +1 t_zV_r,rt,Course,Role <= +2
 triangle1_Course,Role,User,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Role,User
  <= +2
 triangle1_Course,Role,CourseSession,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,CourseSession
  +1 t_zV_r,rt,Role,CourseSession <= +2
 triangle1_Course,Role,Clinic,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,Clinic +1 t_zV_r,rt,Role,Clinic
  <= +2
 triangle1_Course,Role,Location,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,Location +1 t_zV_r,rt,Role,Location
  <= +2
 triangle1_Course,Role,Therapist,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,Therapist +1 t_zV_r,rt,Role,Therapist
  <= +2
 triangle1_Course,Role,Instructor,r: +1 t_zV_r,rt,Course,Role +1 t_zV_r,lb,Course,Instructor +1 t_zV_r,rt,Role,Instructor
  <= +2
 axisVdifRT_c,Course,Role: -52 t_zV_c,rt,Course,Role -1 t_x_c,Role +1 t_x_c,Course >= -42
 triangle1_CertificateTemplate,Course,Role,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,Role
  +1 t_zV_c,rt,Course,Role <= +2
 triangle1_Course,Role,User,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Role,User
  <= +2
 triangle1_Course,Role,Location,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,Location +1 t_zV_c,rt,Role,Location
  <= +2
 triangle1_Course,Role,Therapist,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,Therapist +1 t_zV_c,rt,Role,Therapist
  <= +2
 triangle1_Course,Role,Instructor,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,Instructor +1 t_zV_c,rt,Role,Instructor
  <= +2
 axisVdifLB_r,Course,Role: -48 t_zV_r,lb,Course,Role +1 t_x_r,Role -1 t_x_r,Course >= -42
 triangle2_Client,Course,Role,r: +1 t_zV_r,lb,Client,Course +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Course,Role
  <= +2
 triangle2_CertificateTemplate,Course,Role,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,Role
  +1 t_zV_r,lb,Course,Role <= +2
 triangle2_Course,Role,User,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,User +1 t_zV_r,lb,Role,User
  <= +2
 triangle2_Course,Role,Clinic,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,Clinic +1 t_zV_r,lb,Role,Clinic
  <= +2
 triangle2_Course,Role,Location,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,Location +1 t_zV_r,lb,Role,Location
  <= +2
 triangle2_Course,Role,Therapist,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,Therapist +1 t_zV_r,lb,Role,Therapist
  <= +2
 axisVdifLB_c,Role,User: -52 t_zV_c,lb,Role,User +1 t_x_c,User -1 t_x_c,Role >= -42
 axisVdifLB_c,Role,Therapist: -52 t_zV_c,lb,Role,Therapist +1 t_x_c,Therapist -1 t_x_c,Role >= -42
 axisVdifRT_c,CertificateTemplate,Role: -52 t_zV_c,rt,CertificateTemplate,Role -1 t_x_c,Role +1 t_x_c,CertificateTemplate
  >= -42
 axisVdifRT_c,Course,User: -53 t_zV_c,rt,Course,User -1 t_x_c,User +1 t_x_c,Course >= -42
 axisVdifRT_c,Course,Location: -53 t_zV_c,rt,Course,Location -1 t_x_c,Location +1 t_x_c,Course >= -42
 axisVdifRT_c,Course,Therapist: -53 t_zV_c,rt,Course,Therapist -1 t_x_c,Therapist +1 t_x_c,Course
  >= -42
 axisVdifRT_r,Course,Role: -48 t_zV_r,rt,Course,Role -1 t_x_r,Role +1 t_x_r,Course >= -42
 choose1_Course,Role: +1 t_zV_r,rt,Course,Role +1 t_zV_c,rt,Course,Role +1 t_zV_r,lb,Course,Role +1 t_zV_c,lb,Course,Role
  = +1
 triangle2_Client,Role,Therapist,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,Therapist +1 t_zV_r,lb,Role,Therapist
  <= +2
 triangle2_Client,Role,Location,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,Location +1 t_zV_r,lb,Role,Location
  <= +2
 triangle2_Client,Role,Clinic,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,Clinic +1 t_zV_r,lb,Role,Clinic
  <= +2
 triangle2_Client,Role,CourseSession,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,CourseSession
  +1 t_zV_r,lb,Role,CourseSession <= +2
 triangle2_Client,Role,User,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Role,User
  <= +2
 triangle1_Client,Course,Role,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Course,Role
  <= +2
 triangle1_Client,CertificateTemplate,Role,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Role
  +1 t_zV_r,rt,CertificateTemplate,Role <= +2
 axisVdifLB_r,Client,Role: -51 t_zV_r,lb,Client,Role +1 t_x_r,Role -1 t_x_r,Client >= -42
 triangle2_Course,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,Course,Clinic +1 t_zV_c,rt,Course,Instructor
  <= +2
 triangle2_Course,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,Course,Clinic +1 t_zV_c,rt,Course,Therapist
  <= +2
 triangle2_Course,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,Course,Clinic +1 t_zV_c,rt,Course,Location
  <= +2
 triangle2_Client,Course,Clinic,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,Course,Clinic
  <= +2
 triangle1_Course,User,Clinic,c: +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Course,Clinic +1 t_zV_c,rt,User,Clinic
  <= +2
 triangle1_Course,Role,Clinic,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,Clinic +1 t_zV_c,rt,Role,Clinic
  <= +2
 axisVdifLB_c,Course,Clinic: -52 t_zV_c,lb,Course,Clinic +1 t_x_c,Clinic -1 t_x_c,Course >= -42
 triangle1_Location,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,lb,Location,Instructor
  +1 t_zV_r,rt,Location,Therapist <= +2
 triangle1_Clinic,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,lb,Clinic,Therapist
  +1 t_zV_r,rt,Clinic,Location <= +2
 triangle1_CourseSession,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,CourseSession,Location
  +1 t_zV_r,lb,CourseSession,Therapist <= +2
 triangle1_User,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,User,Location +1 t_zV_r,lb,User,Therapist
  <= +2
 triangle1_Role,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,Role,Location +1 t_zV_r,lb,Role,Therapist
  <= +2
 triangle1_Course,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,Course,Location
  +1 t_zV_r,lb,Course,Therapist <= +2
 triangle1_Client,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,Client,Location
  +1 t_zV_r,lb,Client,Therapist <= +2
 axisVdifRT_r,Location,Therapist: -52 t_zV_r,rt,Location,Therapist -1 t_x_r,Therapist +1 t_x_r,Location
  >= -42
 triangle2_Course,Role,Instructor,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,Instructor +1 t_zV_r,lb,Role,Instructor
  <= +2
 triangle2_CertificateTemplate,Role,Instructor,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,Instructor
  +1 t_zV_r,lb,Role,Instructor <= +2
 triangle2_Client,Role,Instructor,r: +1 t_zV_r,lb,Client,Role +1 t_zV_r,rt,Client,Instructor +1 t_zV_r,lb,Role,Instructor
  <= +2
 triangle1_Role,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,Role,Therapist
  +1 t_zV_r,lb,Role,Instructor <= +2
 triangle1_Role,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,Role,Location
  +1 t_zV_r,lb,Role,Instructor <= +2
 triangle1_Role,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,Role,Clinic +1 t_zV_r,lb,Role,Instructor
  <= +2
 triangle1_Role,CourseSession,Instructor,r: +1 t_zV_r,rt,Role,CourseSession +1 t_zV_r,lb,Role,Instructor
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle1_Role,User,Instructor,r: +1 t_zV_r,rt,Role,User +1 t_zV_r,lb,Role,Instructor +1 t_zV_r,rt,User,Instructor
  <= +2
 axisVdifLB_r,Role,Instructor: -48 t_zV_r,lb,Role,Instructor +1 t_x_r,Instructor -1 t_x_r,Role >= -42
 axisVdifRT_r,CertificateTemplate,User: -49 t_zV_r,rt,CertificateTemplate,User -1 t_x_r,User +1 t_x_r,CertificateTemplate
  >= -42
 triangle1_Client,CertificateTemplate,User,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,User
  +1 t_zV_r,rt,CertificateTemplate,User <= +2
 triangle1_CertificateTemplate,User,CourseSession,r: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,User,CourseSession <= +2
 triangle1_CertificateTemplate,User,Clinic,r: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,User,Clinic <= +2
 triangle1_CertificateTemplate,User,Therapist,r: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_r,lb,CertificateTemplate,Therapist
  +1 t_zV_r,rt,User,Therapist <= +2
 triangle1_CertificateTemplate,User,Instructor,r: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_r,lb,CertificateTemplate,Instructor
  +1 t_zV_r,rt,User,Instructor <= +2
 triangle2_CertificateTemplate,Course,User,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,User
  +1 t_zV_r,lb,Course,User <= +2
 triangle2_CertificateTemplate,Role,User,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,User
  +1 t_zV_r,lb,Role,User <= +2
 triangle1_CertificateTemplate,Course,User,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,User
  +1 t_zV_c,rt,Course,User <= +2
 triangle1_CertificateTemplate,Role,User,c: +1 t_zV_c,rt,CertificateTemplate,Role +1 t_zV_c,lb,CertificateTemplate,User
  +1 t_zV_c,rt,Role,User <= +2
 triangle2_Client,CertificateTemplate,User,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,User
  +1 t_zV_c,lb,CertificateTemplate,User <= +2
 triangle2_CertificateTemplate,User,CourseSession,c: +1 t_zV_c,lb,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,User,CourseSession <= +2
 triangle2_CertificateTemplate,User,Clinic,c: +1 t_zV_c,lb,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_c,lb,User,Clinic <= +2
 triangle2_CertificateTemplate,User,Instructor,c: +1 t_zV_c,lb,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,Instructor
  +1 t_zV_c,lb,User,Instructor <= +2
 triangle2_Client,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,Client,Location
  +1 t_zV_c,rt,Client,Therapist <= +2
 triangle2_Course,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,Course,Location
  +1 t_zV_c,rt,Course,Therapist <= +2
 triangle2_Role,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,Role,Location +1 t_zV_c,rt,Role,Therapist
  <= +2
 triangle1_Client,User,Location,c: +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Client,Location +1 t_zV_c,rt,User,Location
  <= +2
 triangle1_Course,User,Location,c: +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Course,Location +1 t_zV_c,rt,User,Location
  <= +2
 triangle1_Role,User,Location,c: +1 t_zV_c,rt,Role,User +1 t_zV_c,lb,Role,Location +1 t_zV_c,rt,User,Location
  <= +2
 triangle1_User,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,User,Location
  +1 t_zV_c,lb,User,Instructor <= +2
 triangle2_User,CourseSession,Location,c: +1 t_zV_c,lb,User,CourseSession +1 t_zV_c,rt,User,Location
  +1 t_zV_c,lb,CourseSession,Location <= +2
 triangle2_User,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,User,Clinic +1 t_zV_c,rt,User,Location
  <= +2
 triangle2_CourseSession,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,CourseSession,Location
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle2_Clinic,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,rt,Clinic,Therapist
  +1 t_zV_c,lb,Clinic,Location <= +2
 triangle2_Location,Therapist,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,lb,Location,Therapist
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 axisVdifLB_c,CertificateTemplate,User: -54 t_zV_c,lb,CertificateTemplate,User +1 t_x_c,User -1 t_x_c,CertificateTemplate
  >= -42
 triangle1_Client,CertificateTemplate,User,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,User
  +1 t_zV_c,rt,CertificateTemplate,User <= +2
 triangle1_CertificateTemplate,User,CourseSession,c: +1 t_zV_c,rt,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,User,CourseSession <= +2
 triangle1_CertificateTemplate,User,Clinic,c: +1 t_zV_c,rt,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,User,Clinic <= +2
 triangle1_Client,CertificateTemplate,Therapist,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Therapist
  +1 t_zV_c,rt,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,CertificateTemplate,Therapist
  +1 t_zV_c,lb,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Role,Therapist,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,Therapist
  +1 t_zV_c,lb,Role,Therapist <= +2
 triangle2_CertificateTemplate,CourseSession,Therapist,c: +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,CertificateTemplate,Therapist +1 t_zV_c,lb,CourseSession,Therapist <= +2
 triangle2_CertificateTemplate,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,CertificateTemplate,Therapist <= +2
 triangle2_CertificateTemplate,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,User,Location,c: +1 t_zV_c,rt,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,Location
  +1 t_zV_c,rt,User,Location <= +2
 triangle1_CertificateTemplate,User,Instructor,c: +1 t_zV_c,rt,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,Instructor
  +1 t_zV_c,rt,User,Instructor <= +2
 triangle2_CertificateTemplate,Role,User,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,User
  +1 t_zV_c,lb,Role,User <= +2
 choose1_CertificateTemplate,User: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,User
  +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,User = +1
 triangle2_CertificateTemplate,CourseSession,Location,r: +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,CertificateTemplate,Location +1 t_zV_r,lb,CourseSession,Location <= +2
 triangle2_CertificateTemplate,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,CertificateTemplate,Clinic
  +1 t_zV_r,rt,CertificateTemplate,Location <= +2
 axisVdifRT_c,CertificateTemplate,Location: -53 t_zV_c,rt,CertificateTemplate,Location -1 t_x_c,Location
  +1 t_x_c,CertificateTemplate >= -42
 triangle1_Client,CertificateTemplate,Location,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Location
  +1 t_zV_c,rt,CertificateTemplate,Location <= +2
 triangle1_CertificateTemplate,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_c,lb,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_c,lb,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Role,Location,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_c,lb,Role,Location <= +2
 triangle2_CertificateTemplate,User,Location,c: +1 t_zV_c,lb,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_c,lb,User,Location <= +2
 triangle2_CertificateTemplate,CourseSession,Location,c: +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,CertificateTemplate,Location +1 t_zV_c,lb,CourseSession,Location <= +2
 triangle2_CertificateTemplate,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,CertificateTemplate,Clinic
  +1 t_zV_c,rt,CertificateTemplate,Location <= +2
 axisVdifLB_r,CertificateTemplate,Location: -49 t_zV_r,lb,CertificateTemplate,Location +1 t_x_r,Location
  -1 t_x_r,CertificateTemplate >= -42
 triangle1_CertificateTemplate,Course,Location,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,Course,Location <= +2
 triangle1_CertificateTemplate,Role,Location,r: +1 t_zV_r,rt,CertificateTemplate,Role +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,Role,Location <= +2
 triangle1_CertificateTemplate,User,Location,r: +1 t_zV_r,rt,CertificateTemplate,User +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,User,Location <= +2
 triangle1_CertificateTemplate,CourseSession,Location,r: +1 t_zV_r,rt,CertificateTemplate,CourseSession
  +1 t_zV_r,lb,CertificateTemplate,Location +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle1_CertificateTemplate,Clinic,Location,r: +1 t_zV_r,rt,CertificateTemplate,Clinic +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,Clinic,Location <= +2
 triangle2_Client,CertificateTemplate,Location,r: +1 t_zV_r,lb,Client,CertificateTemplate +1 t_zV_r,rt,Client,Location
  +1 t_zV_r,lb,CertificateTemplate,Location <= +2
 triangle2_CertificateTemplate,Location,Therapist,r: +1 t_zV_r,lb,Location,Therapist +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,CertificateTemplate,Therapist <= +2
 triangle2_CertificateTemplate,Location,Instructor,r: +1 t_zV_r,lb,Location,Instructor +1 t_zV_r,lb,CertificateTemplate,Location
  +1 t_zV_r,rt,CertificateTemplate,Instructor <= +2
 axisVdifLB_c,Location,Therapist: -53 t_zV_c,lb,Location,Therapist +1 t_x_c,Therapist -1 t_x_c,Location
  >= -42
 axisVdifRT_c,User,Location: -53 t_zV_c,rt,User,Location -1 t_x_c,Location +1 t_x_c,User >= -42
 axisVdifRT_c,CertificateTemplate,User: -53 t_zV_c,rt,CertificateTemplate,User -1 t_x_c,User +1 t_x_c,CertificateTemplate
  >= -42
 axisVdifRT_c,CertificateTemplate,Therapist: -53 t_zV_c,rt,CertificateTemplate,Therapist -1 t_x_c,Therapist
  +1 t_x_c,CertificateTemplate >= -42
 axisVdifRT_r,CertificateTemplate,Location: -51 t_zV_r,rt,CertificateTemplate,Location -1 t_x_r,Location
  +1 t_x_r,CertificateTemplate >= -42
 triangle1_Client,CertificateTemplate,Location,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Location
  +1 t_zV_r,rt,CertificateTemplate,Location <= +2
 triangle1_CertificateTemplate,Location,Therapist,r: +1 t_zV_r,rt,Location,Therapist +1 t_zV_r,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,CertificateTemplate,Therapist <= +2
 triangle1_CertificateTemplate,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,CertificateTemplate,Instructor <= +2
 triangle2_CertificateTemplate,Course,Location,r: +1 t_zV_r,lb,CertificateTemplate,Course +1 t_zV_r,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,Course,Location <= +2
 triangle2_CertificateTemplate,Role,Location,r: +1 t_zV_r,lb,CertificateTemplate,Role +1 t_zV_r,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,Role,Location <= +2
 triangle2_CertificateTemplate,User,Location,r: +1 t_zV_r,lb,CertificateTemplate,User +1 t_zV_r,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,User,Location <= +2
 choose1_CertificateTemplate,Location: +1 t_zV_r,rt,CertificateTemplate,Location +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_r,lb,CertificateTemplate,Location +1 t_zV_c,lb,CertificateTemplate,Location = +1
 DistSumLB_Course,CourseSession: +1 t_dist_r,Course,CourseSession +1 t_dist_c,Course,CourseSession
  >= +7.5
 triangle2_Client,Clinic,Instructor,r: +1 t_zV_r,lb,Clinic,Instructor +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,Client,Instructor
  <= +2
 triangle2_Client,Clinic,Therapist,r: +1 t_zV_r,lb,Clinic,Therapist +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,Client,Therapist
  <= +2
 triangle2_Client,Clinic,Location,r: +1 t_zV_r,lb,Clinic,Location +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,Client,Location
  <= +2
 triangle1_Client,CourseSession,Clinic,r: +1 t_zV_r,rt,Client,CourseSession +1 t_zV_r,lb,Client,Clinic
  +1 t_zV_r,rt,CourseSession,Clinic <= +2
 triangle1_Client,User,Clinic,r: +1 t_zV_r,rt,Client,User +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,User,Clinic
  <= +2
 triangle1_Client,Role,Clinic,r: +1 t_zV_r,rt,Client,Role +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,Role,Clinic
  <= +2
 triangle1_Client,Course,Clinic,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,Clinic +1 t_zV_r,rt,Course,Clinic
  <= +2
 triangle1_Client,CertificateTemplate,Clinic,r: +1 t_zV_r,rt,Client,CertificateTemplate +1 t_zV_r,lb,Client,Clinic
  +1 t_zV_r,rt,CertificateTemplate,Clinic <= +2
 axisVdifLB_r,Client,Clinic: -51 t_zV_r,lb,Client,Clinic +1 t_x_r,Clinic -1 t_x_r,Client >= -42
 DistAxisLB1_r,Course,CourseSession: +7.5 t_zV_c,rt,Course,CourseSession +7.5 t_zV_c,lb,Course,CourseSession
  +1 t_dist_r,Course,CourseSession >= +7.5
 triangle1_Client,Course,CourseSession,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,CourseSession
  +1 t_zV_c,rt,Course,CourseSession <= +2
 triangle1_CertificateTemplate,Course,CourseSession,c: +1 t_zV_c,rt,CertificateTemplate,Course +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,Course,CourseSession <= +2
 triangle1_Course,CourseSession,Clinic,c: +1 t_zV_c,rt,Course,CourseSession +1 t_zV_c,lb,Course,Clinic
  +1 t_zV_c,rt,CourseSession,Clinic <= +2
 triangle1_Course,CourseSession,Location,c: +1 t_zV_c,rt,Course,CourseSession +1 t_zV_c,lb,Course,Location
  +1 t_zV_c,rt,CourseSession,Location <= +2
 triangle1_Course,CourseSession,Therapist,c: +1 t_zV_c,rt,Course,CourseSession +1 t_zV_c,lb,Course,Therapist
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle2_Course,Role,CourseSession,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,CourseSession
  +1 t_zV_c,lb,Role,CourseSession <= +2
 triangle2_Course,User,CourseSession,c: +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Course,CourseSession
  +1 t_zV_c,lb,User,CourseSession <= +2
 axisVdifLB_c,Course,CourseSession: -52 t_zV_c,lb,Course,CourseSession +1 t_x_c,CourseSession -1 t_x_c,Course
  >= -42
 triangle1_Course,Role,CourseSession,c: +1 t_zV_c,rt,Course,Role +1 t_zV_c,lb,Course,CourseSession
  +1 t_zV_c,rt,Role,CourseSession <= +2
 triangle1_Course,User,CourseSession,c: +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Course,CourseSession
  +1 t_zV_c,rt,User,CourseSession <= +2
 triangle2_Client,Course,CourseSession,c: +1 t_zV_c,lb,Client,Course +1 t_zV_c,rt,Client,CourseSession
  +1 t_zV_c,lb,Course,CourseSession <= +2
 triangle2_Course,CourseSession,Clinic,c: +1 t_zV_c,lb,Course,CourseSession +1 t_zV_c,rt,Course,Clinic
  +1 t_zV_c,lb,CourseSession,Clinic <= +2
 triangle2_Course,CourseSession,Location,c: +1 t_zV_c,lb,Course,CourseSession +1 t_zV_c,rt,Course,Location
  +1 t_zV_c,lb,CourseSession,Location <= +2
 triangle2_Course,CourseSession,Therapist,c: +1 t_zV_c,lb,Course,CourseSession +1 t_zV_c,rt,Course,Therapist
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 axisVdifRT_c,Course,CourseSession: -56 t_zV_c,rt,Course,CourseSession -1 t_x_c,CourseSession +1 t_x_c,Course
  >= -42
 DistAxisLB1_c,Course,CourseSession: +12 t_zV_r,rt,Course,CourseSession +12 t_zV_r,lb,Course,CourseSession
  +1 t_dist_c,Course,CourseSession >= +12
 choose1_Course,CourseSession: +1 t_zV_r,rt,Course,CourseSession +1 t_zV_c,rt,Course,CourseSession
  +1 t_zV_r,lb,Course,CourseSession +1 t_zV_c,lb,Course,CourseSession = +1
 triangle2_Course,User,CourseSession,r: +1 t_zV_r,lb,Course,User +1 t_zV_r,rt,Course,CourseSession
  +1 t_zV_r,lb,User,CourseSession <= +2
 triangle2_Course,Role,CourseSession,r: +1 t_zV_r,lb,Course,Role +1 t_zV_r,rt,Course,CourseSession
  +1 t_zV_r,lb,Role,CourseSession <= +2
 triangle1_Course,CourseSession,Instructor,r: +1 t_zV_r,rt,Course,CourseSession +1 t_zV_r,lb,Course,Instructor
  +1 t_zV_r,rt,CourseSession,Instructor <= +2
 triangle1_Course,CourseSession,Therapist,r: +1 t_zV_r,rt,Course,CourseSession +1 t_zV_r,lb,Course,Therapist
  +1 t_zV_r,rt,CourseSession,Therapist <= +2
 triangle1_Course,CourseSession,Location,r: +1 t_zV_r,rt,Course,CourseSession +1 t_zV_r,lb,Course,Location
  +1 t_zV_r,rt,CourseSession,Location <= +2
 triangle1_Course,CourseSession,Clinic,r: +1 t_zV_r,rt,Course,CourseSession +1 t_zV_r,lb,Course,Clinic
  +1 t_zV_r,rt,CourseSession,Clinic <= +2
 triangle1_CertificateTemplate,Course,CourseSession,r: +1 t_zV_r,rt,CertificateTemplate,Course +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,Course,CourseSession <= +2
 triangle1_Client,Course,CourseSession,r: +1 t_zV_r,rt,Client,Course +1 t_zV_r,lb,Client,CourseSession
  +1 t_zV_r,rt,Course,CourseSession <= +2
 axisVdifRT_r,Course,CourseSession: -51 t_zV_r,rt,Course,CourseSession -1 t_x_r,CourseSession +1 t_x_r,Course
  >= -42
 triangle2_Client,Role,Instructor,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,Instructor +1 t_zV_c,lb,Role,Instructor
  <= +2
 triangle2_Client,Role,Therapist,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,Therapist +1 t_zV_c,lb,Role,Therapist
  <= +2
 triangle2_Client,Role,Location,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,Location +1 t_zV_c,lb,Role,Location
  <= +2
 triangle2_Client,Role,CourseSession,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,CourseSession
  +1 t_zV_c,lb,Role,CourseSession <= +2
 triangle2_Client,Role,User,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Role,User
  <= +2
 triangle1_Client,Course,Role,c: +1 t_zV_c,rt,Client,Course +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Course,Role
  <= +2
 triangle1_Client,CertificateTemplate,Role,c: +1 t_zV_c,rt,Client,CertificateTemplate +1 t_zV_c,lb,Client,Role
  +1 t_zV_c,rt,CertificateTemplate,Role <= +2
 axisVdifLB_c,Client,Role: -53 t_zV_c,lb,Client,Role +1 t_x_c,Role -1 t_x_c,Client >= -42
 DistSumLB_CourseSession,Instructor: +1 t_dist_r,CourseSession,Instructor +1 t_dist_c,CourseSession,Instructor
  >= +9.5
 triangle2_CertificateTemplate,Course,Instructor,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Instructor
  +1 t_zV_c,lb,Course,Instructor <= +2
 triangle2_CertificateTemplate,Course,Therapist,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Therapist
  +1 t_zV_c,lb,Course,Therapist <= +2
 triangle2_CertificateTemplate,Course,Location,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Location
  +1 t_zV_c,lb,Course,Location <= +2
 triangle2_CertificateTemplate,Course,Clinic,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_c,lb,Course,Clinic <= +2
 triangle2_CertificateTemplate,Course,CourseSession,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,Course,CourseSession <= +2
 triangle2_CertificateTemplate,Course,User,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,User
  +1 t_zV_c,lb,Course,User <= +2
 triangle2_CertificateTemplate,Course,Role,c: +1 t_zV_c,lb,CertificateTemplate,Course +1 t_zV_c,rt,CertificateTemplate,Role
  +1 t_zV_c,lb,Course,Role <= +2
 triangle2_Client,CertificateTemplate,Course,c: +1 t_zV_c,lb,Client,CertificateTemplate +1 t_zV_c,rt,Client,Course
  +1 t_zV_c,lb,CertificateTemplate,Course <= +2
 axisVdifLB_c,CertificateTemplate,Course: -54 t_zV_c,lb,CertificateTemplate,Course +1 t_x_c,Course
  -1 t_x_c,CertificateTemplate >= -42
 axisVdifLB_c,CourseSession,Instructor: -56 t_zV_c,lb,CourseSession,Instructor -1 t_x_c,CourseSession
  +1 t_x_c,Instructor >= -42
 triangle1_CourseSession,Clinic,Instructor,c: +1 t_zV_c,rt,Clinic,Instructor +1 t_zV_c,rt,CourseSession,Clinic
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle1_CourseSession,Location,Instructor,c: +1 t_zV_c,rt,Location,Instructor +1 t_zV_c,rt,CourseSession,Location
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle1_CourseSession,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,CourseSession,Therapist
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle2_Client,CourseSession,Instructor,c: +1 t_zV_c,lb,Client,CourseSession +1 t_zV_c,rt,Client,Instructor
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle2_CertificateTemplate,CourseSession,Instructor,c: +1 t_zV_c,lb,CertificateTemplate,CourseSession
  +1 t_zV_c,rt,CertificateTemplate,Instructor +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle2_Course,CourseSession,Instructor,c: +1 t_zV_c,lb,Course,CourseSession +1 t_zV_c,rt,Course,Instructor
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle2_Role,CourseSession,Instructor,c: +1 t_zV_c,lb,Role,CourseSession +1 t_zV_c,rt,Role,Instructor
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 triangle2_User,CourseSession,Instructor,c: +1 t_zV_c,lb,User,CourseSession +1 t_zV_c,rt,User,Instructor
  +1 t_zV_c,lb,CourseSession,Instructor <= +2
 axisVdifRT_c,CourseSession,Instructor: -54 t_zV_c,rt,CourseSession,Instructor +1 t_x_c,CourseSession
  -1 t_x_c,Instructor >= -42
 DistAxisLB1_r,CourseSession,Instructor: +9.5 t_zV_c,rt,CourseSession,Instructor +9.5 t_zV_c,lb,CourseSession,Instructor
  +1 t_dist_r,CourseSession,Instructor >= +9.5
 triangle1_Client,CourseSession,Instructor,c: +1 t_zV_c,rt,Client,CourseSession +1 t_zV_c,lb,Client,Instructor
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle1_CertificateTemplate,CourseSession,Instructor,c: +1 t_zV_c,rt,CertificateTemplate,CourseSession
  +1 t_zV_c,lb,CertificateTemplate,Instructor +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle1_Course,CourseSession,Instructor,c: +1 t_zV_c,rt,Course,CourseSession +1 t_zV_c,lb,Course,Instructor
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle1_Role,CourseSession,Instructor,c: +1 t_zV_c,rt,Role,CourseSession +1 t_zV_c,lb,Role,Instructor
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle1_User,CourseSession,Instructor,c: +1 t_zV_c,rt,User,CourseSession +1 t_zV_c,lb,User,Instructor
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle2_CourseSession,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,CourseSession,Clinic
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle2_CourseSession,Location,Instructor,c: +1 t_zV_c,lb,Location,Instructor +1 t_zV_c,lb,CourseSession,Location
  +1 t_zV_c,rt,CourseSession,Instructor <= +2
 triangle2_CourseSession,Therapist,Instructor,c: +1 t_zV_c,lb,CourseSession,Therapist +1 t_zV_c,rt,CourseSession,Instructor
  +1 t_zV_c,lb,Therapist,Instructor <= +2
 axisVdifLB_r,CourseSession,Instructor: -51 t_zV_r,lb,CourseSession,Instructor +1 t_x_r,Instructor
  -1 t_x_r,CourseSession >= -42
 triangle1_CourseSession,Clinic,Instructor,r: +1 t_zV_r,rt,Clinic,Instructor +1 t_zV_r,rt,CourseSession,Clinic
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle1_CourseSession,Location,Instructor,r: +1 t_zV_r,rt,Location,Instructor +1 t_zV_r,rt,CourseSession,Location
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle1_CourseSession,Therapist,Instructor,r: +1 t_zV_r,rt,Therapist,Instructor +1 t_zV_r,rt,CourseSession,Therapist
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle2_Client,CourseSession,Instructor,r: +1 t_zV_r,lb,Client,CourseSession +1 t_zV_r,rt,Client,Instructor
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle2_CertificateTemplate,CourseSession,Instructor,r: +1 t_zV_r,lb,CertificateTemplate,CourseSession
  +1 t_zV_r,rt,CertificateTemplate,Instructor +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle2_Course,CourseSession,Instructor,r: +1 t_zV_r,lb,Course,CourseSession +1 t_zV_r,rt,Course,Instructor
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle2_Role,CourseSession,Instructor,r: +1 t_zV_r,lb,Role,CourseSession +1 t_zV_r,rt,Role,Instructor
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 triangle2_User,CourseSession,Instructor,r: +1 t_zV_r,lb,User,CourseSession +1 t_zV_r,rt,User,Instructor
  +1 t_zV_r,lb,CourseSession,Instructor <= +2
 DistAxisLB1_c,CourseSession,Instructor: +13 t_zV_r,rt,CourseSession,Instructor +13 t_zV_r,lb,CourseSession,Instructor
  +1 t_dist_c,CourseSession,Instructor >= +13
 choose1_CourseSession,Instructor: +1 t_zV_r,rt,CourseSession,Instructor +1 t_zV_c,rt,CourseSession,Instructor
  +1 t_zV_r,lb,CourseSession,Instructor +1 t_zV_c,lb,CourseSession,Instructor = +1
 DistSumLB_User,Therapist: +1 t_dist_r,User,Therapist +1 t_dist_c,User,Therapist >= +8.5
 triangle1_Client,User,Therapist,c: +1 t_zV_c,rt,Client,User +1 t_zV_c,lb,Client,Therapist +1 t_zV_c,rt,User,Therapist
  <= +2
 triangle1_CertificateTemplate,User,Therapist,c: +1 t_zV_c,rt,CertificateTemplate,User +1 t_zV_c,lb,CertificateTemplate,Therapist
  +1 t_zV_c,rt,User,Therapist <= +2
 triangle1_Course,User,Therapist,c: +1 t_zV_c,rt,Course,User +1 t_zV_c,lb,Course,Therapist +1 t_zV_c,rt,User,Therapist
  <= +2
 triangle1_Role,User,Therapist,c: +1 t_zV_c,rt,Role,User +1 t_zV_c,lb,Role,Therapist +1 t_zV_c,rt,User,Therapist
  <= +2
 triangle1_User,Therapist,Instructor,c: +1 t_zV_c,rt,Therapist,Instructor +1 t_zV_c,rt,User,Therapist
  +1 t_zV_c,lb,User,Instructor <= +2
 triangle2_User,CourseSession,Therapist,c: +1 t_zV_c,lb,User,CourseSession +1 t_zV_c,rt,User,Therapist
  +1 t_zV_c,lb,CourseSession,Therapist <= +2
 triangle2_User,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,User,Clinic +1 t_zV_c,rt,User,Therapist
  <= +2
 triangle2_User,Location,Therapist,c: +1 t_zV_c,lb,Location,Therapist +1 t_zV_c,lb,User,Location +1 t_zV_c,rt,User,Therapist
  <= +2
 DistAxisLB1_r,User,Therapist: +8.5 t_zV_c,rt,User,Therapist +8.5 t_zV_c,lb,User,Therapist +1 t_dist_r,User,Therapist
  >= +8.5
 triangle1_User,CourseSession,Therapist,c: +1 t_zV_c,rt,User,CourseSession +1 t_zV_c,lb,User,Therapist
  +1 t_zV_c,rt,CourseSession,Therapist <= +2
 triangle1_User,Clinic,Therapist,c: +1 t_zV_c,rt,Clinic,Therapist +1 t_zV_c,rt,User,Clinic +1 t_zV_c,lb,User,Therapist
  <= +2
 triangle1_User,Location,Therapist,c: +1 t_zV_c,rt,Location,Therapist +1 t_zV_c,rt,User,Location +1 t_zV_c,lb,User,Therapist
  <= +2
 triangle2_Client,User,Therapist,c: +1 t_zV_c,lb,Client,User +1 t_zV_c,rt,Client,Therapist +1 t_zV_c,lb,User,Therapist
  <= +2
 triangle2_CertificateTemplate,User,Therapist,c: +1 t_zV_c,lb,CertificateTemplate,User +1 t_zV_c,rt,CertificateTemplate,Therapist
  +1 t_zV_c,lb,User,Therapist <= +2
 triangle2_Course,User,Therapist,c: +1 t_zV_c,lb,Course,User +1 t_zV_c,rt,Course,Therapist +1 t_zV_c,lb,User,Therapist
  <= +2
 triangle2_Role,User,Therapist,c: +1 t_zV_c,lb,Role,User +1 t_zV_c,rt,Role,Therapist +1 t_zV_c,lb,User,Therapist
  <= +2
 triangle2_User,Therapist,Instructor,c: +1 t_zV_c,lb,User,Therapist +1 t_zV_c,rt,User,Instructor +1 t_zV_c,lb,Therapist,Instructor
  <= +2
 compDistAxis2_c,User,Therapist: -1 t_x_c,Therapist +1 t_x_c,User +1 t_dist_c,User,Therapist >= +0
 compDistAxis1_c,User,Therapist: +1 t_x_c,Therapist -1 t_x_c,User +1 t_dist_c,User,Therapist >= +0
 axisVdifRT_c,User,Therapist: -53 t_zV_c,rt,User,Therapist -1 t_x_c,Therapist +1 t_x_c,User >= -42
 axisVdifLB_c,User,Therapist: -53 t_zV_c,lb,User,Therapist +1 t_x_c,Therapist -1 t_x_c,User >= -42
 centerDistAxis2_c,User: -1 t_x_c,User +1 t_center_dist_c,User >= -16.5
 centerDistAxis1_c,User: +1 t_x_c,User +1 t_center_dist_c,User >= +16.5
 triangle2_Role,Clinic,Instructor,c: +1 t_zV_c,lb,Clinic,Instructor +1 t_zV_c,lb,Role,Clinic +1 t_zV_c,rt,Role,Instructor
  <= +2
 triangle2_Role,Clinic,Therapist,c: +1 t_zV_c,lb,Clinic,Therapist +1 t_zV_c,lb,Role,Clinic +1 t_zV_c,rt,Role,Therapist
  <= +2
 triangle2_Role,Clinic,Location,c: +1 t_zV_c,lb,Clinic,Location +1 t_zV_c,lb,Role,Clinic +1 t_zV_c,rt,Role,Location
  <= +2
 triangle2_Course,Role,Clinic,c: +1 t_zV_c,lb,Course,Role +1 t_zV_c,rt,Course,Clinic +1 t_zV_c,lb,Role,Clinic
  <= +2
 triangle2_CertificateTemplate,Role,Clinic,c: +1 t_zV_c,lb,CertificateTemplate,Role +1 t_zV_c,rt,CertificateTemplate,Clinic
  +1 t_zV_c,lb,Role,Clinic <= +2
 triangle2_Client,Role,Clinic,c: +1 t_zV_c,lb,Client,Role +1 t_zV_c,rt,Client,Clinic +1 t_zV_c,lb,Role,Clinic
  <= +2
 triangle1_Role,CourseSession,Clinic,c: +1 t_zV_c,rt,Role,CourseSession +1 t_zV_c,lb,Role,Clinic +1 t_zV_c,rt,CourseSession,Clinic
  <= +2
 triangle1_Role,User,Clinic,c: +1 t_zV_c,rt,Role,User +1 t_zV_c,lb,Role,Clinic +1 t_zV_c,rt,User,Clinic
  <= +2
 axisVdifLB_c,Role,Clinic: -52 t_zV_c,lb,Role,Clinic +1 t_x_c,Clinic -1 t_x_c,Role >= -42
 centerDistAxis2_c,Therapist: -1 t_x_c,Therapist +1 t_center_dist_c,Therapist >= -16.5
 centerDistAxis1_c,Therapist: +1 t_x_c,Therapist +1 t_center_dist_c,Therapist >= +16.5
 newcut_1: +1.0 t_zV_r,lb,Client,User -0.06249999999999999 t_x_r,User +0.06249999999999999 t_x_r,Client -0.06249999999999999 t_dist_r,Client,User -0.06249999999999999 t_dist_c,Client,User <= -0.06249999999999999
 newcut_2: +1.0 t_zV_r,lb,CourseSession,Therapist -0.029411764705882353 t_x_r,Therapist +0.058823529411764705 t_x_r,CourseSession -0.029411764705882353 t_x_r,Course -0.0980392156862745 t_dist_c,Course,CourseSession -0.029411764705882353 t_center_dist_r,Therapist -0.029411764705882353 t_center_dist_r,Course -0.1568627450980392 t_dist_r,Course,CourseSession <= -0.7647058823529411
 newcut_3: +1.0 t_zV_r,lb,CourseSession,Instructor +0.014084507042253523 t_x_c,Location -0.02816901408450704 t_x_c,CourseSession +0.01408450704225352 t_x_c,CertificateTemplate +0.01408450704225352 t_x_c,Client -0.05633802816901408 t_x_r,Instructor +0.08450704225352113 t_x_r,Location +0.05633802816901408 t_x_r,CourseSession -0.05633802816901408 t_x_r,User -0.02816901408450704 t_x_r,Client -0.014084507042253523 t_x_c,Instructor -0.02816901408450704 t_center_dist_r,Clinic -0.02816901408450704 t_dist_r,Role,User -0.02816901408450704 t_dist_r,Clinic,Location -0.02816901408450704 t_dist_c,User,Therapist -0.02816901408450704 t_center_dist_c,Location -0.02816901408450704 t_dist_c,Clinic,Location -0.10060362173038227 t_dist_r,Location,CourseSession -0.02816901408450704 t_center_dist_r,Location -0.05633802816901408 t_dist_r,Client,User -0.02816901408450704 t_dist_c,Course,CourseSession -0.02816901408450704 t_center_dist_c,Clinic -0.02816901408450704 t_dist_c,Clinic,Therapist -0.10060362173038227 t_dist_c,Location,CourseSession -0.16197183098591547 t_dist_r,Client,Instructor -0.02816901408450704 t_center_dist_c,Role -0.02816901408450704 t_center_dist_r,Instructor -0.02816901408450704 t_center_dist_c,CertificateTemplate -0.02816901408450704 t_center_dist_c,Course -0.02816901408450704 t_dist_c,Role,User -0.02816901408450704 t_center_dist_c,CourseSession -0.02816901408450704 t_center_dist_r,Therapist -0.08450704225352113 t_dist_c,Client,Location -0.08450704225352113 t_dist_r,Client,Location -0.02816901408450704 t_dist_r,Instructor,Location -0.02816901408450704 t_center_dist_r,Course -0.02816901408450704 t_center_dist_c,User -0.01408450704225352 t_dist_c,Client,CertificateTemplate -0.02816901408450704 t_center_dist_r,Client -0.02816901408450704 t_center_dist_r,CertificateTemplate -0.02816901408450704 t_center_dist_r,Role -0.02816901408450704 t_center_dist_c,Client -0.16197183098591547 t_dist_c,Client,Instructor -0.05633802816901408 t_dist_c,Client,User -0.02816901408450704 t_center_dist_c,Instructor -0.02816901408450704 t_dist_r,Course,CourseSession -0.02816901408450704 t_center_dist_r,User -0.02816901408450704 t_center_dist_c,Therapist -0.04225352112676056 t_dist_c,Instructor,Location -0.02816901408450704 t_center_dist_r,CourseSession <= -3.711770623742454
 newcut_4: +1.0 t_zV_r,lb,Role,User +1.0 t_zV_r,lb,User,Instructor -0.07007299270072989 t_x_r,Instructor +0.07007299270072989 t_x_r,Location -0.07153284671532843 t_x_r,User +0.07153284671532843 t_x_r,Role -0.07153284671532843 t_dist_r,Role,User -0.07007299270072989 t_dist_r,Client,User -0.07153284671532843 t_dist_c,Role,User -0.07007299270072989 t_dist_c,Client,Location -0.07007299270072989 t_dist_r,Client,Location -0.07007299270072989 t_dist_r,Instructor,Location -0.07007299270072989 t_dist_c,Client,User -0.07007299270072989 t_dist_c,Instructor,Location <= -0.12043795620437941
 newcut_5: +0.6507936507936511 t_zV_c,lb,Course,Role +0.6507936507936511 t_zV_c,lb,Role,Therapist -0.03174603174603175 t_x_c,Therapist -0.015873015873015876 t_x_c,CourseSession +0.015873015873015876 t_x_c,User +0.03174603174603175 t_x_c,Course -0.03174603174603175 t_dist_c,User,Therapist -0.08465608465608467 t_dist_c,Course,CourseSession -0.015873015873015876 t_center_dist_c,Role -0.015873015873015876 t_dist_c,Role,User -0.015873015873015876 t_center_dist_c,CourseSession -0.08465608465608467 t_dist_r,Course,CourseSession <= 0.05555555555555589
 newcut_6: +0.6933947772657456 t_zV_c,lb,Course,User +0.6933947772657454 t_zV_c,lb,User,Clinic -0.03195084485407067 t_x_c,Clinic -0.016282642089093708 t_x_c,CourseSession +0.01566820276497696 t_x_c,User +0.032565284178187416 t_x_c,Course -0.03195084485407067 t_dist_c,User,Therapist -0.08684075780849976 t_dist_c,Course,CourseSession -0.03195084485407067 t_dist_c,Clinic,Therapist -0.016282642089093708 t_center_dist_c,Role -0.016282642089093708 t_dist_c,Role,User -0.016282642089093708 t_center_dist_c,CourseSession -0.08684075780849976 t_dist_r,Course,CourseSession <= 0.0827956989247317
 newcut_7: +0.8314465408805032 t_zV_c,lb,Role,CourseSession +0.8314465408805032 t_zV_c,lb,CourseSession,Clinic -0.03270440251572327 t_x_c,Clinic -0.0012578616352201233 t_x_c,CourseSession +0.016352201257861635 t_x_c,User +0.017610062893081757 t_x_c,Role -0.03270440251572327 t_dist_c,User,Therapist -0.03270440251572327 t_dist_c,Clinic,Therapist -0.0012578616352201277 t_center_dist_c,Role -0.016352201257861635 t_dist_c,Role,User -0.0012578616352201277 t_center_dist_c,CourseSession <= 0.8421383647798741
 newcut_8: +1.0 t_zV_c,lb,CourseSession,Clinic -0.04 t_x_c,Clinic +0.04 t_x_c,CourseSession -0.04 t_dist_c,Clinic,Location -0.04 t_dist_c,Location,CourseSession <= -0.06
 newcut_9: +1.0 t_zV_c,lb,User,CourseSession -0.03999999999999999 t_x_c,Clinic -0.03999999999999999 t_x_c,CourseSession +0.07999999999999997 t_x_c,User -0.07999999999999997 t_dist_c,User,Therapist -0.03999999999999999 t_dist_c,Clinic,Location -0.07999999999999997 t_dist_c,Clinic,Therapist -0.03999999999999999 t_dist_c,Location,CourseSession <= 0.059999999999999984
 newcut_10: -0.14103186702402326 t_zV_c,lb,Location,Instructor -0.14103186702402326 t_zV_r,lb,Location,Instructor -0.14103186702402326 t_zV_c,rt,Location,Instructor -0.14103186702402326 t_zV_r,rt,Location,Instructor +1.0 t_zV_r,lb,Course,User +0.007422729843369645 t_x_c,Location -0.022268189530108937 t_x_c,CourseSession +0.01484545968673929 t_x_c,CertificateTemplate -0.01484545968673929 t_x_r,Instructor -0.0038174039194472445 t_x_r,Therapist +0.01484545968673929 t_x_r,Location +0.01484545968673929 t_x_r,CourseSession -0.06659249059480196 t_x_r,User +0.04071897514077063 t_x_r,Course -0.01484545968673929 t_center_dist_r,Clinic -0.01484545968673929 t_dist_r,Role,User -0.022056111534584092 t_dist_c,User,Therapist -0.01484545968673929 t_center_dist_c,Location -0.1249846320293098 t_dist_r,Therapist,CourseSession -0.03859819518552216 t_dist_r,CertificateTemplate,CourseSession -0.01484545968673929 t_center_dist_r,Location -0.01484545968673929 t_dist_r,Client,User -0.022056111534584092 t_dist_r,User,Therapist -0.07366175711229687 t_dist_c,Course,CourseSession -0.01484545968673929 t_center_dist_c,Clinic -0.007422729843369645 t_dist_c,Location,CourseSession -0.01484545968673929 t_center_dist_c,Role -0.01484545968673929 t_center_dist_r,Instructor -0.01484545968673929 t_center_dist_c,CertificateTemplate -0.01484545968673929 t_center_dist_c,Course -0.00919004647274337 t_dist_c,Role,User -0.01484545968673929 t_center_dist_c,CourseSession -0.025873515454031337 t_center_dist_r,Therapist -0.01484545968673929 t_dist_c,Client,Location -0.01484545968673929 t_dist_r,Instructor,Location -0.04071897514077063 t_center_dist_r,Course -0.01484545968673929 t_center_dist_c,User -0.03859819518552216 t_dist_c,CertificateTemplate,CourseSession -0.01484545968673929 t_center_dist_r,Client -0.01484545968673929 t_center_dist_r,CertificateTemplate -0.01484545968673929 t_center_dist_r,Role -0.01484545968673929 t_center_dist_c,Client -0.01484545968673929 t_dist_c,Client,Instructor -0.01484545968673929 t_dist_c,Client,User -0.01484545968673929 t_center_dist_c,Instructor -0.05881629742555758 t_dist_r,Course,CourseSession -0.01484545968673929 t_center_dist_r,User -0.01484545968673929 t_center_dist_c,Therapist -0.01226364061078463 t_dist_c,Instructor,Location -0.1249846320293098 t_dist_c,Therapist,CourseSession -0.01484545968673929 t_center_dist_r,CourseSession <= -2.498455518945635
 newcut_11: +1.0 t_zV_c,lb,Course,Role +1.0 t_zV_c,lb,Role,Location -0.02439024390243903 t_x_c,Location -0.02439024390243903 t_x_c,CourseSession +0.04878048780487806 t_x_c,Course -0.04878048780487806 t_dist_c,Course,CourseSession -0.02439024390243903 t_dist_c,Location,CourseSession <= 1.0609756097560978
 newcut_12: -0.7333333333333327 t_zV_r,rt,CertificateTemplate,CourseSession -0.7333333333333327 t_zV_c,rt,CertificateTemplate,CourseSession -0.7333333333333327 t_zV_r,lb,CertificateTemplate,CourseSession -0.7333333333333327 t_zV_c,lb,CertificateTemplate,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Instructor +0.04166666666666666 t_x_c,Location -0.04166666666666666 t_x_c,CourseSession +0.08333333333333331 t_x_c,CertificateTemplate -0.08333333333333331 t_x_c,Instructor -0.21666666666666662 t_dist_r,CertificateTemplate,CourseSession -0.04166666666666666 t_dist_c,Location,CourseSession -0.0833333333333333 t_dist_c,Client,Location -0.21666666666666662 t_dist_c,CertificateTemplate,CourseSession -0.0833333333333333 t_dist_c,Client,Instructor <= -2.4041666666666655
 newcut_13: +1.0 t_zV_c,lb,Client,Course +1.0 t_zV_c,lb,Course,Location -0.00045351473922902713 t_x_c,Location -0.00045351473922902713 t_x_c,CourseSession +0.0009070294784580543 t_x_c,Course -0.0009070294784580543 t_dist_c,Course,CourseSession -0.00045351473922902713 t_dist_c,Location,CourseSession -0.04716553287981859 t_dist_c,Client,Location <= 1.0011337868480725
 newcut_14: +1.0 t_zV_c,lb,Client,Therapist +1.0 t_zV_c,lb,Therapist,Instructor +0.04444444444444444 t_x_c,Client -0.04444444444444444 t_x_c,Instructor -0.04444444444444443 t_dist_c,Client,Instructor <= 1.0222222222222224
 newcut_15: -0.06811101083358446 t_zV_c,lb,Location,Instructor -0.06811101083358446 t_zV_r,lb,Location,Instructor -0.06811101083358446 t_zV_c,rt,Location,Instructor -0.06811101083358446 t_zV_r,rt,Location,Instructor +0.5371550173296424 t_zV_r,rt,Clinic,Instructor +0.537155017329642 t_zV_r,lb,Client,Instructor +0.0032949795202435986 t_x_c,Location -0.009884938560730796 t_x_c,CourseSession +0.006589959040487197 t_x_c,CertificateTemplate -0.00716958008774573 t_x_r,Instructor -0.001694560896125281 t_x_r,Therapist +0.007169580087745733 t_x_r,Location -0.011485357184849117 t_x_r,Clinic +0.006589959040487197 t_x_r,CourseSession -0.02297071436969823 t_x_r,User -0.004895398144361919 t_x_r,Course +0.02297071436969823 t_x_r,Client -0.018075316225336313 t_center_dist_r,Clinic -0.006589959040487197 t_dist_r,Role,User -0.009790796288723836 t_dist_c,User,Therapist -0.006589959040487197 t_center_dist_c,Location -0.055481178969435066 t_dist_r,Therapist,CourseSession -0.017133893505266715 t_dist_r,CertificateTemplate,CourseSession -0.006589959040487197 t_center_dist_r,Location -0.02297071436969823 t_dist_r,Client,User -0.009790796288723836 t_dist_r,User,Therapist -0.03269874914375077 t_dist_c,Course,CourseSession -0.006589959040487197 t_center_dist_c,Clinic -0.0032949795202435986 t_dist_c,Location,CourseSession -0.006589959040487197 t_center_dist_c,Role -0.006589959040487197 t_center_dist_r,Instructor -0.006589959040487197 t_center_dist_c,CertificateTemplate -0.006589959040487197 t_center_dist_c,Course -0.004079498453634932 t_dist_c,Role,User -0.006589959040487197 t_center_dist_c,CourseSession -0.011485357184849117 t_center_dist_r,Therapist -0.006589959040487197 t_dist_c,Client,Location -0.0005796210472585357 t_dist_r,Client,Location -0.0071695800877457335 t_dist_r,Instructor,Location -0.011485357184849115 t_center_dist_r,Course -0.006589959040487197 t_center_dist_c,User -0.017133893505266715 t_dist_c,CertificateTemplate,CourseSession -0.006589959040487197 t_center_dist_r,Client -0.006589959040487197 t_center_dist_r,CertificateTemplate -0.006589959040487197 t_center_dist_r,Role -0.006589959040487197 t_center_dist_c,Client -0.006589959040487197 t_dist_c,Client,Instructor -0.02297071436969823 t_dist_c,Client,User -0.006589959040487197 t_center_dist_c,Instructor -0.026108790103263568 t_dist_r,Course,CourseSession -0.006589959040487197 t_center_dist_r,User -0.006589959040487197 t_center_dist_c,Therapist -0.005922696594224736 t_dist_c,Instructor,Location -0.055481178969435066 t_dist_c,Therapist,CourseSession -0.006589959040487197 t_center_dist_r,CourseSession <= -0.9009102024781166
 newcut_16: +1.0 t_zV_c,lb,Client,Clinic -1.0 t_zV_r,rt,Client,Instructor +0.02385164399438784 t_x_c,Location -0.02385164399438784 t_x_c,Clinic -0.049574005164806104 t_x_r,Instructor +0.049574005164806104 t_x_r,Location -0.02385164399438784 t_dist_c,Clinic,Location -0.049574005164806104 t_dist_c,Client,Location -0.049574005164806104 t_dist_r,Client,Location -0.049574005164806104 t_dist_r,Instructor,Location -0.21984993594827054 t_dist_c,Client,Instructor -0.04095243904918765 t_dist_c,Instructor,Location <= -0.4418552634254457
 newcut_17: +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Clinic -0.023705004389815622 t_x_c,Clinic -0.023705004389815622 t_x_c,CourseSession +0.0017559262510974541 t_x_c,Course +0.045654082528533785 t_x_c,CertificateTemplate -0.023705004389815622 t_dist_c,Clinic,Location -0.11870061457418785 t_dist_r,CertificateTemplate,CourseSession -0.0017559262510974585 t_dist_c,Course,CourseSession -0.023705004389815622 t_dist_c,Location,CourseSession -0.11870061457418785 t_dist_c,CertificateTemplate,CourseSession <= 0.0640035118525026
 newcut_18: +1.0 t_zV_c,lb,Client,Role +1.0 t_zV_c,lb,Role,Clinic +0.023582766439909295 t_x_c,Location -0.024036281179138322 t_x_c,Clinic +0.00045351473922902713 t_x_c,Role -0.024036281179138322 t_dist_c,Clinic,Location -0.00045351473922902713 t_dist_c,Location,CourseSession -0.00045351473922902713 t_center_dist_c,Role -0.00045351473922902713 t_center_dist_c,CourseSession -0.04716553287981859 t_dist_c,Client,Location <= 1.0002267573696144
 newcut_19: +0.7101096224116932 t_zV_c,lb,CertificateTemplate,Role +0.7101096224116934 t_zV_c,lb,Role,Instructor +0.01644336175395859 t_x_c,Location -0.015834348355663826 t_x_c,CourseSession +0.0006090133982947631 t_x_c,Role +0.03166869671132765 t_x_c,CertificateTemplate -0.03288672350791718 t_x_c,Instructor -0.0823386114494519 t_dist_r,CertificateTemplate,CourseSession -0.01644336175395859 t_dist_c,Location,CourseSession -0.0006090133982947644 t_center_dist_c,Role -0.0006090133982947644 t_center_dist_c,CourseSession -0.03288672350791718 t_dist_c,Client,Location -0.0823386114494519 t_dist_c,CertificateTemplate,CourseSession -0.03288672350791718 t_dist_c,Client,Instructor <= 0.07606577344701614
 newcut_20: +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_c,lb,User,Location -0.02243456584960531 t_x_c,Location -0.00041545492314083943 t_x_c,Clinic -0.02201911092646447 t_x_c,CourseSession +0.0008309098462816799 t_x_c,User +0.04403822185292894 t_x_c,CertificateTemplate -0.000830909846281679 t_dist_c,User,Therapist -0.00041545492314083943 t_dist_c,Clinic,Location -0.11449937681761525 t_dist_r,CertificateTemplate,CourseSession -0.000830909846281679 t_dist_c,Clinic,Therapist -0.02201911092646447 t_dist_c,Location,CourseSession -0.11449937681761525 t_dist_c,CertificateTemplate,CourseSession <= 0.0950145409223105
 newcut_21: +1.0 t_zV_c,lb,Therapist,Instructor +0.0869565217391304 t_x_c,Therapist +0.0434782608695652 t_x_c,Location -0.0434782608695652 t_x_c,Clinic -0.0869565217391304 t_x_c,Instructor -0.0434782608695652 t_dist_c,Clinic,Location -0.0869565217391304 t_dist_c,Clinic,Therapist -0.0869565217391304 t_dist_c,Client,Location -0.0869565217391304 t_dist_c,Client,Instructor <= 0.0434782608695652
 newcut_22: +1.0 t_zV_c,lb,Role,CourseSession -0.04166666666666668 t_x_c,CourseSession +0.04166666666666668 t_x_c,Role -0.04166666666666668 t_center_dist_c,Role -0.04166666666666668 t_center_dist_c,CourseSession <= 0.08333333333333336
 newcut_23: -0.06225865189699949 t_zV_c,lb,Location,Instructor -0.062258651896999494 t_zV_r,lb,Location,Instructor -0.06225865189699949 t_zV_c,rt,Location,Instructor -0.062258651896999494 t_zV_r,rt,Location,Instructor -0.49926323530608996 t_zV_r,lb,Clinic,Location -0.024895319712458067 t_zV_r,rt,CourseSession,Therapist -0.024895319712458067 t_zV_c,rt,CourseSession,Therapist -0.024895319712458067 t_zV_r,lb,CourseSession,Therapist -0.024895319712458067 t_zV_c,lb,CourseSession,Therapist +0.0032767711524736573 t_x_c,Location -0.009830313457420973 t_x_c,CourseSession +0.006553542304947315 t_x_c,CertificateTemplate -0.006553542304947315 t_x_r,Instructor -0.0016851965927007374 t_x_r,Therapist +0.03595086064428241 t_x_r,Location -0.01142188801719389 t_x_r,Clinic +0.006553542304947315 t_x_r,CourseSession -0.022843776034387784 t_x_r,User -0.004868345712246578 t_x_r,Course -0.006553542304947315 t_x_r,Client -0.017975430322141207 t_center_dist_r,Clinic -0.006553542304947315 t_dist_r,Role,User -0.009736691424493155 t_dist_c,User,Therapist -0.006553542304947315 t_center_dist_c,Location -0.13629665775509908 t_dist_c,Clinic,Location -0.05517458473879453 t_dist_r,Therapist,CourseSession -0.017039209992863017 t_dist_r,CertificateTemplate,CourseSession -0.006553542304947315 t_center_dist_r,Location -0.02284377603438778 t_dist_r,Client,User -0.009736691424493155 t_dist_r,User,Therapist -0.032518052770262394 t_dist_c,Course,CourseSession -0.006553542304947315 t_center_dist_c,Clinic -0.0032767711524736573 t_dist_c,Location,CourseSession -0.006553542304947315 t_center_dist_c,Role -0.006553542304947315 t_center_dist_r,Instructor -0.006553542304947315 t_center_dist_c,CertificateTemplate -0.006553542304947315 t_center_dist_c,Course -0.00405695476020548 t_dist_c,Role,User -0.006553542304947315 t_center_dist_c,CourseSession -0.011421888017193892 t_center_dist_r,Therapist -0.03595086064428241 t_dist_c,Client,Location -0.029397318339335097 t_dist_r,Client,Location -0.006553542304947315 t_dist_r,Instructor,Location -0.011421888017193894 t_center_dist_r,Course -0.006553542304947315 t_center_dist_c,User -0.017039209992863017 t_dist_c,CertificateTemplate,CourseSession -0.006553542304947315 t_center_dist_r,Client -0.006553542304947315 t_center_dist_r,CertificateTemplate -0.006553542304947315 t_center_dist_r,Role -0.006553542304947315 t_center_dist_c,Client -0.006553542304947315 t_dist_c,Client,Instructor -0.02284377603438778 t_dist_c,Client,User -0.006553542304947315 t_center_dist_c,Instructor -0.025964510465315078 t_dist_r,Course,CourseSession -0.006553542304947315 t_center_dist_r,User -0.006553542304947315 t_center_dist_c,Therapist -0.005413795817130391 t_dist_c,Instructor,Location -0.05517458473879453 t_dist_c,Therapist,CourseSession -0.006553542304947315 t_center_dist_r,CourseSession <= -1.9436632808959493
 newcut_24: -1.0 t_zV_r,lb,Location,Therapist +1.0 t_zV_c,lb,CertificateTemplate,Location +0.19576129392080308 t_x_c,Location -0.19576129392080308 t_x_c,Clinic -0.07975460122699383 t_x_c,CourseSession +0.07975460122699383 t_x_c,CertificateTemplate +0.08282208588957053 t_x_r,Therapist -0.08282208588957053 t_x_r,User -0.08282208588957053 t_dist_c,User,Therapist -0.19576129392080308 t_dist_c,Clinic,Location -0.08282208588957053 t_dist_r,Client,User -0.08282208588957053 t_dist_r,User,Therapist -0.39152258784160615 t_dist_c,Clinic,Therapist -0.39152258784160615 t_dist_c,Location,Therapist -0.08282208588957053 t_dist_c,Client,Location -0.08282208588957053 t_dist_r,Client,Location -0.07975460122699383 t_dist_c,CertificateTemplate,CourseSession -0.08282208588957053 t_dist_c,Client,User -0.28483786152497803 t_dist_c,Client,CourseSession -0.28483786152497803 t_dist_r,Client,CourseSession <= -4.015994741454863
 newcut_25: -0.0012009554712401715 t_zV_c,lb,Location,Instructor -0.0012009554712401713 t_zV_r,lb,Location,Instructor -0.0012009554712401715 t_zV_c,rt,Location,Instructor -0.0012009554712401713 t_zV_r,rt,Location,Instructor +0.4587816484606293 t_zV_r,lb,CertificateTemplate,CourseSession +0.4587816484606291 t_zV_r,lb,CourseSession,Clinic +0.026294604001890064 t_zV_r,rt,CourseSession,Therapist +0.026294604001890064 t_zV_r,lb,CourseSession,Therapist -0.00025283273078740453 t_x_c,CourseSession +0.00012641636539370226 t_x_c,CertificateTemplate +0.00012641636539370226 t_x_c,Client -0.00012641636539370226 t_x_r,Instructor -0.00026770524436313414 t_x_r,Therapist +0.00012641636539370226 t_x_r,Location -0.014238572684564217 t_x_r,Clinic -0.0003792490961811068 t_x_r,User -0.000710162523241092 t_x_r,Course -0.014364989049957923 t_center_dist_r,Clinic -0.00012641636539370226 t_dist_r,Role,User -0.00037924909618110677 t_dist_c,User,Therapist -0.00012641636539370226 t_center_dist_c,Location -0.002767853052830533 t_dist_r,Therapist,CourseSession -0.00012641636539370226 t_center_dist_r,Location -0.00012641636539370226 t_dist_r,Client,User -0.00037924909618110677 t_dist_r,User,Therapist -0.0041246437650023626 t_dist_c,Course,CourseSession -0.00012641636539370226 t_center_dist_c,Clinic -0.00012641636539370226 t_center_dist_c,Role -0.00012641636539370226 t_center_dist_r,Instructor -0.00012641636539370226 t_center_dist_c,CertificateTemplate -0.00012641636539370226 t_center_dist_c,Course -7.82577500056252e-05 t_dist_c,Role,User -0.00012641636539370226 t_center_dist_c,CourseSession -0.0007733707059379431 t_center_dist_r,Therapist -0.00012641636539370226 t_dist_r,Instructor,Location -0.0008365788886347942 t_center_dist_r,Course -0.00012641636539370226 t_center_dist_c,User -0.00012641636539370226 t_dist_c,CertificateTemplate,CourseSession -0.00012641636539370226 t_center_dist_r,Client -0.029771054050216923 t_center_dist_r,CertificateTemplate -0.00012641636539370226 t_center_dist_r,Role -0.00012641636539370226 t_center_dist_c,Client -0.00012641636539370226 t_dist_c,Client,Instructor -0.00012641636539370226 t_dist_c,Client,User -0.00012641636539370226 t_center_dist_c,Instructor -0.0041246437650023626 t_dist_r,Course,CourseSession -0.0004514870192632224 t_dist_c,Client,CourseSession -0.00012641636539370226 t_center_dist_r,User -0.00012641636539370226 t_center_dist_c,Therapist -0.0001044309105426236 t_dist_c,Instructor,Location -0.0004514870192632224 t_dist_r,Client,CourseSession -0.002767853052830533 t_dist_c,Therapist,CourseSession -0.00012641636539370226 t_center_dist_r,CourseSession <= 0.12110717441972035
 newcut_26: -0.4219208609631481 t_zV_r,rt,Client,CourseSession -0.42192086096314807 t_zV_c,rt,Client,CourseSession -0.4219208609631481 t_zV_r,lb,Client,CourseSession -0.42192086096314807 t_zV_c,lb,Client,CourseSession -0.08532177410588106 t_zV_r,rt,Role,User -0.08532177410588106 t_zV_c,rt,Role,User -0.08532177410588106 t_zV_r,lb,Role,User -0.08532177410588106 t_zV_c,lb,Role,User +0.33153603652570934 t_zV_r,rt,CourseSession,Therapist +0.33153603652570934 t_zV_r,lb,CourseSession,Therapist +1.0 t_zV_r,rt,Clinic,Location -0.026252853571040326 t_x_c,CourseSession +0.013126426785520163 t_x_c,CertificateTemplate +0.013126426785520163 t_x_c,Client -0.013126426785520163 t_x_r,Instructor -0.003375366887705185 t_x_r,Therapist +0.07200782693771061 t_x_r,Location -0.02287748668333514 t_x_r,Clinic +0.013126426785520165 t_x_r,CourseSession -0.04575497336667028 t_x_r,User -0.00975105989781498 t_x_r,Course -0.013126426785520166 t_x_r,Client -0.0360039134688553 t_center_dist_r,Clinic -0.013126426785520163 t_dist_r,Role,User -0.019502119795629956 t_dist_c,User,Therapist -0.013126426785520163 t_center_dist_c,Location -0.03489853016060098 t_dist_r,Therapist,CourseSession -0.013126426785520163 t_center_dist_r,Location -0.04575497336667028 t_dist_r,Client,User -0.019502119795629956 t_dist_r,User,Therapist -0.06513207957386673 t_dist_c,Course,CourseSession -0.013126426785520163 t_center_dist_c,Clinic -0.013126426785520163 t_center_dist_c,Role -0.013126426785520163 t_center_dist_r,Instructor -0.013126426785520163 t_center_dist_c,CertificateTemplate -0.013126426785520163 t_center_dist_c,Course -0.008125883248179148 t_dist_c,Role,User -0.013126426785520163 t_center_dist_c,CourseSession -0.02287748668333514 t_center_dist_r,Therapist -0.05888140015219045 t_dist_c,Client,Location -0.05888140015219045 t_dist_r,Client,Location -0.013126426785520163 t_dist_r,Instructor,Location -0.02287748668333514 t_center_dist_r,Course -0.013126426785520163 t_center_dist_c,User -0.013126426785520163 t_dist_c,CertificateTemplate,CourseSession -0.013126426785520163 t_center_dist_r,Client -0.013126426785520163 t_center_dist_r,CertificateTemplate -0.013126426785520163 t_center_dist_r,Role -0.013126426785520163 t_center_dist_c,Client -0.013126426785520163 t_dist_c,Client,Instructor -0.04575497336667028 t_dist_c,Client,User -0.013126426785520163 t_center_dist_c,Instructor -0.05200565278834656 t_dist_r,Course,CourseSession -0.04688009566257201 t_dist_c,Client,CourseSession -0.013126426785520163 t_center_dist_r,User -0.013126426785520163 t_center_dist_c,Therapist -0.010843569953255788 t_dist_c,Instructor,Location -0.04688009566257201 t_dist_r,Client,CourseSession -0.03489853016060098 t_dist_c,Therapist,CourseSession -0.013126426785520163 t_center_dist_r,CourseSession <= -2.3149228176975756
 newcut_27: +1.0 t_zV_c,lb,Client,Course -0.028470201189421868 t_zV_r,rt,Client,CourseSession -0.02847020118942186 t_zV_c,rt,Client,CourseSession -0.028470201189421868 t_zV_r,lb,Client,CourseSession -0.02847020118942186 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,Course,Instructor -0.0008857395925597914 t_x_c,CourseSession +0.000885739592559792 t_x_c,Course +0.04694419840566871 t_x_c,Client -0.04694419840566871 t_x_c,Instructor -0.0008857395925597914 t_dist_c,Course,CourseSession -0.04694419840566871 t_dist_c,Client,Instructor -0.0031633556877135407 t_dist_c,Client,CourseSession -0.0031633556877135403 t_dist_r,Client,CourseSession <= 0.96830317600911
 newcut_28: +0.6667674629573632 t_zV_c,lb,Client,Role -0.019439284634325555 t_zV_r,rt,Client,CourseSession -0.01943928463432555 t_zV_c,rt,Client,CourseSession -0.019439284634325555 t_zV_r,lb,Client,CourseSession -0.01943928463432555 t_zV_c,lb,Client,CourseSession +0.6667674629573632 t_zV_c,lb,Role,Location -0.00030238887208950863 t_x_c,CourseSession +0.0003023888720895092 t_x_c,Role -0.00030238887208950863 t_center_dist_c,Role -0.00030238887208950863 t_center_dist_c,CourseSession -0.032053220441487756 t_dist_c,Client,Location -0.0021599205149250614 t_dist_c,Client,CourseSession -0.0021599205149250614 t_dist_r,Client,CourseSession <= 0.6284936714328914
 newcut_29: -0.05857268280446532 t_zV_r,rt,Client,CourseSession -0.05857268280446532 t_zV_c,rt,Client,CourseSession -0.05857268280446532 t_zV_r,lb,Client,CourseSession -0.05857268280446532 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Therapist +0.023705004389815622 t_x_c,Location -0.023705004389815622 t_x_c,Clinic -0.04741000877963125 t_x_c,CourseSession +0.0017559262510974587 t_x_c,Course +0.04565408252853379 t_x_c,CertificateTemplate -0.023705004389815622 t_dist_c,Clinic,Location -0.0017559262510974587 t_dist_c,Course,CourseSession -0.047410008779631245 t_dist_c,Clinic,Therapist -0.047410008779631245 t_dist_c,Client,Location -0.04565408252853379 t_dist_c,CertificateTemplate,CourseSession -0.16932145992725448 t_dist_c,Client,CourseSession -0.16932145992725448 t_dist_r,Client,CourseSession <= -0.5332998871190268
 newcut_30: +1.0 t_zV_r,lb,Location,Therapist +0.05555555555555556 t_x_r,Location -0.05555555555555556 t_x_r,User -0.05555555555555556 t_dist_c,User,Therapist -0.05555555555555556 t_dist_r,Client,User -0.05555555555555556 t_dist_r,User,Therapist -0.05555555555555556 t_dist_c,Client,Location -0.05555555555555556 t_dist_r,Client,Location -0.05555555555555556 t_dist_c,Client,User <= -0.5277777777777778
 newcut_31: -0.0018400624584963049 t_zV_c,lb,Location,Instructor -0.001840062458496305 t_zV_r,lb,Location,Instructor -0.0018400624584963049 t_zV_c,rt,Location,Instructor -0.001840062458496305 t_zV_r,rt,Location,Instructor +0.4596537565503329 t_zV_r,lb,Course,CourseSession +0.4596537565503329 t_zV_r,lb,CourseSession,Clinic +0.040287683301813836 t_zV_r,rt,CourseSession,Therapist +0.040287683301813836 t_zV_r,lb,CourseSession,Therapist -0.0003873815702097484 t_x_c,CourseSession +0.0001936907851048742 t_x_c,CertificateTemplate +0.0001936907851048742 t_x_c,Client -0.0001936907851048742 t_x_r,Instructor -0.00041016872139855714 t_x_r,Therapist +0.0001936907851048742 t_x_r,Location -0.014213485554019455 t_x_r,Clinic -0.0005810723553146226 t_x_r,User +0.015107881238180194 t_x_r,Course -0.014407176339124328 t_center_dist_r,Clinic -0.0001936907851048742 t_dist_r,Role,User -0.0005810723553146226 t_dist_c,User,Therapist -0.0001936907851048742 t_center_dist_c,Location -0.004240808768611983 t_dist_r,Therapist,CourseSession -0.0001936907851048742 t_center_dist_r,Location -0.0001936907851048742 t_dist_r,Client,User -0.0005810723553146226 t_dist_r,User,Therapist -0.006319636596362955 t_dist_c,Course,CourseSession -0.0001936907851048742 t_center_dist_c,Clinic -0.0001936907851048742 t_center_dist_c,Role -0.0001936907851048742 t_center_dist_r,Instructor -0.0001936907851048742 t_center_dist_c,CertificateTemplate -0.0001936907851048742 t_center_dist_c,Course -0.0001199038193506364 t_dist_c,Role,User -0.0001936907851048742 t_center_dist_c,CourseSession -0.001184931861818054 t_center_dist_r,Therapist -0.0001936907851048742 t_dist_r,Instructor,Location -0.015301572023285075 t_center_dist_r,Course -0.0001936907851048742 t_center_dist_c,User -0.0001936907851048742 t_dist_c,CertificateTemplate,CourseSession -0.0001936907851048742 t_center_dist_r,Client -0.0001936907851048742 t_center_dist_r,CertificateTemplate -0.0001936907851048742 t_center_dist_r,Role -0.0001936907851048742 t_center_dist_c,Client -0.0001936907851048742 t_dist_c,Client,Instructor -0.0001936907851048742 t_dist_c,Client,User -0.0001936907851048742 t_center_dist_c,Instructor -0.006319636596362955 t_dist_r,Course,CourseSession -0.0006917528039459792 t_dist_c,Client,CourseSession -0.0001936907851048742 t_center_dist_r,User -0.0001936907851048742 t_center_dist_c,Therapist -0.00016000543117359174 t_dist_c,Instructor,Location -0.0006917528039459792 t_dist_r,Client,CourseSession -0.004240808768611983 t_dist_c,Therapist,CourseSession -0.0001936907851048742 t_center_dist_r,CourseSession <= 0.40955677080213915
 newcut_32: +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Role -0.024280575539568344 t_x_c,CourseSession -0.02428057553956834 t_x_c,Role +0.0017985611510791416 t_x_c,Course +0.04676258992805754 t_x_c,CertificateTemplate -0.0017985611510791416 t_dist_c,Course,CourseSession -0.02428057553956834 t_center_dist_c,Role -0.02428057553956834 t_center_dist_c,CourseSession -0.04676258992805754 t_dist_c,CertificateTemplate,CourseSession <= 1.0017985611510791
 newcut_33: +1.0 t_zV_c,lb,Client,Course -0.029154518950437462 t_zV_r,rt,Client,CourseSession -0.02915451895043746 t_zV_c,rt,Client,CourseSession -0.029154518950437462 t_zV_r,lb,Client,CourseSession -0.02915451895043746 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,Course,User +0.024036281179138322 t_x_c,Location -0.024036281179138322 t_x_c,Clinic -0.0009070294784580544 t_x_c,CourseSession +0.0009070294784580552 t_x_c,Course -0.048072562358276644 t_dist_c,User,Therapist -0.024036281179138322 t_dist_c,Clinic,Location -0.0009070294784580544 t_dist_c,Course,CourseSession -0.048072562358276644 t_dist_c,Clinic,Therapist -0.048072562358276644 t_dist_c,Client,Location -0.0032393909944930513 t_dist_c,Client,CourseSession -0.003239390994493051 t_dist_r,Client,CourseSession <= 0.9435050210560415
 newcut_34: +0.9014580191050771 t_zV_c,lb,CertificateTemplate,Location +0.5 t_x_c,Location -0.5 t_x_c,Clinic -0.20370370370370366 t_x_c,CourseSession +0.20370370370370366 t_x_c,CertificateTemplate +0.16145746047706833 t_x_r,Therapist +0.050081001061393224 t_x_r,Location -0.21153846153846154 t_x_r,User -0.21153846153846154 t_dist_c,User,Therapist -0.5 t_dist_c,Clinic,Location -0.21153846153846154 t_dist_r,Client,User -0.21153846153846154 t_dist_r,User,Therapist -1.0 t_dist_c,Clinic,Therapist -1.0 t_dist_c,Location,Therapist -0.21153846153846154 t_dist_c,Client,Location -0.21153846153846154 t_dist_r,Client,Location -0.20370370370370366 t_dist_c,CertificateTemplate,CourseSession -0.21153846153846154 t_dist_c,Client,User -0.7275132275132276 t_dist_c,Client,CourseSession -0.7275132275132276 t_dist_r,Client,CourseSession <= -9.806647872824342
 newcut_35: -0.5 t_x_c,User +0.5 t_x_c,Instructor -1.0 t_dist_c,User,Instructor <= -0.5
 newcut_36: +0.5 t_x_c,Therapist -0.5 t_x_c,CourseSession -1.0 t_dist_c,Therapist,CourseSession <= 0.5
 newcut_37: +0.2705882352941178 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_r,lb,User,Therapist +0.011764705882352943 t_x_c,Location -0.011764705882352943 t_x_c,Clinic -0.023529411764705885 t_x_c,CourseSession +0.023529411764705885 t_x_c,CertificateTemplate -0.04945978391356544 t_dist_c,User,Therapist -0.011764705882352943 t_dist_c,Clinic,Location -0.025930372148859546 t_dist_r,User,Therapist -0.02352941176470589 t_dist_c,Clinic,Therapist -0.02352941176470589 t_dist_c,Client,Location -0.023529411764705885 t_dist_c,CertificateTemplate,CourseSession -0.08403361344537819 t_dist_c,Client,CourseSession -0.08403361344537819 t_dist_r,Client,CourseSession <= 0.046818727490996144
 newcut_38: -0.06225865189699949 t_zV_c,lb,Location,Instructor -0.062258651896999494 t_zV_r,lb,Location,Instructor -0.06225865189699949 t_zV_c,rt,Location,Instructor -0.062258651896999494 t_zV_r,rt,Location,Instructor +0.49926323530608996 t_zV_r,lb,Client,CourseSession +0.49926323530608996 t_zV_r,lb,CourseSession,Clinic -0.16552375421638366 t_zV_c,rt,CourseSession,Therapist -0.16552375421638366 t_zV_c,lb,CourseSession,Therapist -0.01310708460989463 t_x_c,CourseSession +0.006553542304947315 t_x_c,CertificateTemplate +0.006553542304947315 t_x_c,Client -0.006553542304947315 t_x_r,Instructor -0.001685196592700739 t_x_r,Therapist +0.006553542304947315 t_x_r,Location -0.01142188801719389 t_x_r,Clinic +0.006553542304947315 t_x_r,CourseSession -0.02284377603438778 t_x_r,User -0.004868345712246577 t_x_r,Course +0.02284377603438778 t_x_r,Client -0.017975430322141207 t_center_dist_r,Clinic -0.006553542304947315 t_dist_r,Role,User -0.009736691424493153 t_dist_c,User,Therapist -0.006553542304947315 t_center_dist_c,Location -0.017423553075408806 t_dist_r,Therapist,CourseSession -0.006553542304947315 t_center_dist_r,Location -0.022843776034387784 t_dist_r,Client,User -0.009736691424493153 t_dist_r,User,Therapist -0.032518052770262394 t_dist_c,Course,CourseSession -0.006553542304947315 t_center_dist_c,Clinic -0.006553542304947315 t_center_dist_c,Role -0.006553542304947315 t_center_dist_r,Instructor -0.006553542304947315 t_center_dist_c,CertificateTemplate -0.006553542304947315 t_center_dist_c,Course -0.00405695476020548 t_dist_c,Role,User -0.006553542304947315 t_center_dist_c,CourseSession -0.011421888017193892 t_center_dist_r,Therapist -0.006553542304947315 t_dist_r,Instructor,Location -0.01142188801719389 t_center_dist_r,Course -0.006553542304947315 t_center_dist_c,User -0.006553542304947315 t_dist_c,CertificateTemplate,CourseSession -0.006553542304947315 t_center_dist_r,Client -0.006553542304947315 t_center_dist_r,CertificateTemplate -0.006553542304947315 t_center_dist_r,Role -0.006553542304947315 t_center_dist_c,Client -0.006553542304947315 t_dist_c,Client,Instructor -0.022843776034387784 t_dist_c,Client,User -0.006553542304947315 t_center_dist_c,Instructor -0.025964510465315078 t_dist_r,Course,CourseSession -0.023405508231954694 t_dist_c,Client,CourseSession -0.006553542304947315 t_center_dist_r,User -0.006553542304947315 t_center_dist_c,Therapist -0.005413795817130391 t_dist_c,Instructor,Location -0.023405508231954694 t_dist_r,Client,CourseSession -0.017423553075408806 t_dist_c,Therapist,CourseSession -0.006553542304947315 t_center_dist_r,CourseSession <= -0.6310274271851245
 newcut_39: -1.0 t_zV_r,lb,Clinic,Therapist +1.0 t_zV_c,lb,CertificateTemplate,Clinic -0.07224500941686995 t_zV_r,rt,Role,User -0.07224500941686995 t_zV_c,rt,Role,User -0.07224500941686995 t_zV_r,lb,Role,User -0.07224500941686995 t_zV_c,lb,Role,User +0.28072346516269475 t_zV_r,rt,CourseSession,Therapist +0.28072346516269475 t_zV_r,lb,CourseSession,Therapist +0.02400522006020864 t_x_c,Location -0.02400522006020864 t_x_c,Clinic -0.05912505695378189 t_x_c,CourseSession +0.05912505695378189 t_x_c,CertificateTemplate -0.011114616833364609 t_x_r,Instructor +0.03048580617151436 t_x_r,Therapist +0.011114616833364609 t_x_r,Location -0.01937118933814975 t_x_r,Clinic +0.01111461683336461 t_x_r,CourseSession -0.03334385050009383 t_x_r,User -0.00825657250478514 t_x_r,Course -0.030485806171514362 t_center_dist_r,Clinic -0.011114616833364609 t_dist_r,Role,User -0.03334385050009383 t_dist_c,User,Therapist -0.011114616833364609 t_center_dist_c,Location -0.02400522006020864 t_dist_c,Clinic,Location -0.029549838438178398 t_dist_r,Therapist,CourseSession -0.011114616833364609 t_center_dist_r,Location -0.011114616833364609 t_dist_r,Client,User -0.03334385050009383 t_dist_r,User,Therapist -0.05514967019221869 t_dist_c,Course,CourseSession -0.011114616833364609 t_center_dist_c,Clinic -0.23568761513659395 t_dist_c,Clinic,Therapist -0.011114616833364609 t_center_dist_c,Role -0.011114616833364609 t_center_dist_r,Instructor -0.011114616833364609 t_center_dist_c,CertificateTemplate -0.011114616833364609 t_center_dist_c,Course -0.006880477087320948 t_dist_c,Role,User -0.011114616833364609 t_center_dist_c,CourseSession -0.019371189338149748 t_center_dist_r,Therapist -0.04801044012041728 t_dist_c,Client,Location -0.011114616833364609 t_dist_r,Instructor,Location -0.019371189338149748 t_center_dist_r,Course -0.011114616833364609 t_center_dist_c,User -0.05912505695378189 t_dist_c,CertificateTemplate,CourseSession -0.011114616833364609 t_center_dist_r,Client -0.011114616833364609 t_center_dist_r,CertificateTemplate -0.011114616833364609 t_center_dist_r,Role -0.011114616833364609 t_center_dist_c,Client -0.011114616833364609 t_dist_c,Client,Instructor -0.011114616833364609 t_dist_c,Client,User -0.011114616833364609 t_center_dist_c,Instructor -0.044035053358854076 t_dist_r,Course,CourseSession -0.13177079745375953 t_dist_c,Client,CourseSession -0.011114616833364609 t_center_dist_r,User -0.011114616833364609 t_center_dist_c,Therapist -0.009181639992779461 t_dist_c,Instructor,Location -0.13177079745375953 t_dist_r,Client,CourseSession -0.029549838438178398 t_dist_c,Therapist,CourseSession -0.011114616833364609 t_center_dist_r,CourseSession <= -2.360044440934663
 newcut_40: +0.62877030162413 t_zV_r,rt,Clinic,Therapist -0.6287703016241298 t_zV_r,rt,Client,User -0.6287703016241298 t_zV_c,rt,Client,User -0.6287703016241298 t_zV_r,lb,Client,User -0.6287703016241298 t_zV_c,lb,Client,User +0.44072608161594123 t_zV_c,lb,CertificateTemplate,User +0.6287703016241297 t_zV_r,lb,User,Therapist +0.019162003548519175 t_x_c,Location -0.019162003548519175 t_x_c,Clinic -0.03832400709703835 t_x_c,CourseSession +0.030162412993039452 t_x_c,User +0.03832400709703835 t_x_c,CertificateTemplate -0.030162412993039452 t_x_c,Client +0.015661252900232018 t_x_r,Therapist -0.015661252900232018 t_x_r,Clinic -0.015661252900232018 t_center_dist_r,Clinic -0.08055862716316226 t_dist_c,User,Therapist -0.019162003548519175 t_dist_c,Clinic,Location -0.11059551430781132 t_dist_r,Client,User -0.0422346200661239 t_dist_r,User,Therapist -0.03832400709703836 t_dist_c,Clinic,Therapist -0.015661252900232018 t_center_dist_r,Therapist -0.03832400709703836 t_dist_c,Client,Location -0.03832400709703835 t_dist_c,CertificateTemplate,CourseSession -0.11059551430781132 t_dist_c,Client,User -0.13687145391799416 t_dist_c,Client,CourseSession -0.13687145391799416 t_dist_r,Client,CourseSession <= -2.1060709666697313
 newcut_41: -0.5 t_x_c,CourseSession +0.5 t_x_c,Client -1.0 t_dist_c,Client,CourseSession <= 0.5
 newcut_42: -1.0 t_zV_r,lb,Location,Therapist +1.0 t_zV_c,lb,Role,Location +0.20553359683794473 t_x_c,Location -0.20553359683794473 t_x_c,Clinic -0.04347826086956523 t_x_c,CourseSession +0.04347826086956523 t_x_c,Role +0.08695652173913046 t_x_r,Therapist -0.08695652173913046 t_x_r,User -0.08695652173913046 t_dist_c,User,Therapist -0.20553359683794473 t_dist_c,Clinic,Location -0.08695652173913046 t_dist_r,Client,User -0.08695652173913046 t_dist_r,User,Therapist -0.41106719367588945 t_dist_c,Clinic,Therapist -0.41106719367588945 t_dist_c,Location,Therapist -0.04347826086956523 t_center_dist_c,Role -0.04347826086956523 t_center_dist_c,CourseSession -0.08695652173913046 t_dist_c,Client,Location -0.08695652173913046 t_dist_r,Client,Location -0.08695652173913046 t_dist_c,Client,User -0.31055900621118027 t_dist_c,Client,CourseSession -0.31055900621118027 t_dist_r,Client,CourseSession <= -4.316770186335405
 newcut_43: +0.6824445775913723 t_zV_c,lb,CertificateTemplate,Role +0.6824445775913723 t_zV_c,lb,Role,Clinic -0.01617735170760935 t_x_c,Clinic -0.015578190533253445 t_x_c,CourseSession +0.0005991611743559006 t_x_c,Role +0.031156381066506894 t_x_c,CertificateTemplate -0.01617735170760935 t_dist_c,Clinic,Location -0.08100659077291793 t_dist_r,CertificateTemplate,CourseSession -0.01617735170760935 t_dist_c,Location,CourseSession -0.0005991611743559036 t_center_dist_c,Role -0.0005991611743559036 t_center_dist_c,CourseSession -0.08100659077291793 t_dist_c,CertificateTemplate,CourseSession <= 0.04248052726183372
 newcut_44: -0.05857268280446532 t_zV_r,rt,Client,CourseSession -0.05857268280446532 t_zV_c,rt,Client,CourseSession -0.05857268280446532 t_zV_r,lb,Client,CourseSession -0.05857268280446532 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Role +1.0 t_zV_c,lb,Role,Location -0.04653204565408252 t_x_c,CourseSession +0.0008779631255487293 t_x_c,Role +0.04565408252853379 t_x_c,CertificateTemplate -0.0008779631255487293 t_center_dist_c,Role -0.0008779631255487293 t_center_dist_c,CourseSession -0.047410008779631245 t_dist_c,Client,Location -0.04565408252853379 t_dist_c,CertificateTemplate,CourseSession -0.16932145992725448 t_dist_c,Client,CourseSession -0.16932145992725448 t_dist_r,Client,CourseSession <= -0.5350558133701242
 newcut_45: -0.08037245773094855 t_zV_r,rt,Client,CourseSession -0.08037245773094855 t_zV_c,rt,Client,CourseSession -0.08037245773094855 t_zV_r,lb,Client,CourseSession -0.08037245773094855 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Instructor -0.046312178387650074 t_x_c,CourseSession +0.0017152658662092672 t_x_c,Course +0.04459691252144081 t_x_c,CertificateTemplate +0.04631217838765008 t_x_c,Client -0.04631217838765008 t_x_c,Instructor -0.0017152658662092672 t_dist_c,Course,CourseSession -0.04459691252144081 t_dist_c,CertificateTemplate,CourseSession -0.04631217838765008 t_dist_c,Client,Instructor -0.16540063709875027 t_dist_c,Client,CourseSession -0.16540063709875027 t_dist_r,Client,CourseSession <= -0.49779465817201674
 newcut_46: +1.0 t_zV_c,lb,Client,Therapist -1.0 t_zV_r,rt,Client,Instructor +0.02385164399438784 t_x_c,Location -0.02385164399438784 t_x_c,Clinic -0.049574005164806104 t_x_r,Instructor +0.049574005164806104 t_x_r,Location -0.02385164399438784 t_dist_c,Clinic,Location -0.04770328798877568 t_dist_c,Clinic,Therapist -0.049574005164806104 t_dist_c,Client,Location -0.049574005164806104 t_dist_r,Client,Location -0.049574005164806104 t_dist_r,Instructor,Location -0.21984993594827054 t_dist_c,Client,Instructor -0.04095243904918765 t_dist_c,Instructor,Location <= -0.4418552634254457
 newcut_47: -1.5736848718835197 t_zV_c,rt,Client,User +1.0 t_zV_r,lb,Client,User -1.5736848718835197 t_zV_c,lb,Client,User -0.33463794843394346 t_zV_r,rt,Client,CourseSession -0.33463794843394346 t_zV_c,rt,Client,CourseSession -0.33463794843394346 t_zV_r,lb,Client,CourseSession -0.33463794843394346 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_r,lb,CertificateTemplate,Instructor -0.29574124876446534 t_zV_c,rt,CourseSession,Therapist -0.29574124876446534 t_zV_c,lb,CourseSession,Therapist +0.023830415480402958 t_x_c,Location -0.023830415480402958 t_x_c,Clinic -0.059370043298765524 t_x_c,CourseSession +0.04766083096080592 t_x_c,CertificateTemplate +0.0117092123379596 t_x_c,Client -0.05252418105884734 t_x_r,Instructor -0.0030109403154753266 t_x_r,Therapist +0.05252418105884734 t_x_r,Location +0.0117092123379596 t_x_r,CourseSession -0.16085530449271995 t_x_r,User -0.008698272022484275 t_x_r,Course +0.1083311234338726 t_x_r,Client -0.0117092123379596 t_center_dist_r,Clinic -0.0117092123379596 t_dist_r,Role,User -0.06505737500577446 t_dist_c,User,Therapist -0.0117092123379596 t_center_dist_c,Location -0.023830415480402958 t_dist_c,Clinic,Location -0.031130657764680564 t_dist_r,Therapist,CourseSession -0.0117092123379596 t_center_dist_r,Location -0.44175094475727217 t_dist_r,Client,User -0.017396544044968546 t_dist_r,User,Therapist -0.05809999645787574 t_dist_c,Course,CourseSession -0.0117092123379596 t_center_dist_c,Clinic -0.047660830960805915 t_dist_c,Clinic,Therapist -0.0117092123379596 t_center_dist_c,Role -0.0117092123379596 t_center_dist_r,Instructor -0.0117092123379596 t_center_dist_c,CertificateTemplate -0.0117092123379596 t_center_dist_c,Course -0.007248560018736895 t_dist_c,Role,User -0.0117092123379596 t_center_dist_c,CourseSession -0.020407484360443873 t_center_dist_r,Therapist -0.05252418105884734 t_dist_c,Client,Location -0.05252418105884734 t_dist_r,Client,Location -0.04081496872088774 t_dist_r,Instructor,Location -0.020407484360443873 t_center_dist_r,Course -0.0117092123379596 t_center_dist_c,User -0.03595161862284632 t_dist_c,CertificateTemplate,CourseSession -0.0117092123379596 t_center_dist_r,Client -0.06423339339680693 t_center_dist_r,CertificateTemplate -0.0117092123379596 t_center_dist_r,Role -0.0117092123379596 t_center_dist_c,Client -0.0117092123379596 t_dist_c,Client,Instructor -0.44175094475727217 t_dist_c,Client,User -0.0117092123379596 t_center_dist_c,Instructor -0.04639078411991614 t_dist_r,Course,CourseSession -0.21203586892416257 t_dist_c,Client,CourseSession -0.0117092123379596 t_center_dist_r,User -0.0117092123379596 t_center_dist_c,Therapist -0.033716713291168134 t_dist_c,Instructor,Location -0.21203586892416257 t_dist_r,Client,CourseSession -0.031130657764680564 t_dist_c,Therapist,CourseSession -0.0117092123379596 t_center_dist_r,CourseSession <= -6.014991434928012
 newcut_48: -0.11175748925926961 t_zV_c,lb,Location,Instructor -0.11175748925926961 t_zV_r,lb,Location,Instructor -0.11175748925926961 t_zV_c,rt,Location,Instructor -0.11175748925926961 t_zV_r,rt,Location,Instructor -1.0 t_zV_r,rt,Course,CourseSession +1.0 t_zV_c,lb,Course,Location -1.0 t_zV_r,rt,CourseSession,Location -0.053375218884378815 t_x_c,Location -0.011763946237817853 t_x_c,CourseSession +0.053375218884378815 t_x_c,Course +0.011763946237817853 t_x_c,CertificateTemplate -0.011763946237817853 t_x_r,Instructor -0.003899650686569455 t_x_r,Therapist +0.011763946237817853 t_x_r,Location +0.011763946237817853 t_x_r,CourseSession -0.04265784556586252 t_x_r,User +0.03479355001461412 t_x_r,Course -0.011763946237817853 t_x_r,Client -0.011763946237817853 t_center_dist_r,Clinic -0.011763946237817853 t_dist_r,Role,User -0.01912995309022682 t_dist_c,User,Therapist -0.011763946237817853 t_center_dist_c,Location -0.12767745210842207 t_dist_r,Therapist,CourseSession -0.03058626021832642 t_dist_r,CertificateTemplate,CourseSession -0.011763946237817853 t_center_dist_r,Location -0.042657845565862516 t_dist_r,Client,User -0.01912995309022682 t_dist_r,User,Therapist -0.2688448069622363 t_dist_c,Course,CourseSession -0.011763946237817853 t_center_dist_c,Clinic -0.2220409105590159 t_dist_c,Location,CourseSession -0.011763946237817853 t_center_dist_c,Role -0.011763946237817853 t_center_dist_r,Instructor -0.011763946237817853 t_center_dist_c,CertificateTemplate -0.011763946237817853 t_center_dist_c,Course -0.007282442909125337 t_dist_c,Role,User -0.011763946237817853 t_center_dist_c,CourseSession -0.023029603776796276 t_center_dist_r,Therapist -0.06618573804149822 t_dist_c,Client,Location -0.05442179180368036 t_dist_r,Client,Location -0.011763946237817853 t_dist_r,Instructor,Location -0.03479355001461412 t_center_dist_r,Course -0.011763946237817853 t_center_dist_c,User -0.03058626021832642 t_dist_c,CertificateTemplate,CourseSession -0.011763946237817853 t_center_dist_r,Client -0.011763946237817853 t_center_dist_r,CertificateTemplate -0.011763946237817853 t_center_dist_r,Role -0.011763946237817853 t_center_dist_c,Client -0.011763946237817853 t_dist_c,Client,Instructor -0.042657845565862516 t_dist_c,Client,User -0.011763946237817853 t_center_dist_c,Instructor -0.06008350687455158 t_dist_r,Course,CourseSession -0.011763946237817853 t_center_dist_r,User -0.011763946237817853 t_center_dist_c,Therapist -0.009718042544284313 t_dist_c,Instructor,Location -0.12767745210842207 t_dist_c,Therapist,CourseSession -0.011763946237817853 t_center_dist_r,CourseSession <= -3.1518123333458896
 newcut_49: -0.6396142090916069 t_zV_r,rt,Client,CourseSession -0.6396142090916068 t_zV_c,rt,Client,CourseSession -0.6396142090916069 t_zV_r,lb,Client,CourseSession -0.6396142090916068 t_zV_c,lb,Client,CourseSession +4.680238470280938 t_zV_c,lb,Course,User +3.680238470280937 t_zV_c,lb,User,Clinic +0.00994955436364722 t_x_c,Location -0.20571052579107907 t_x_c,Clinic -0.12980280342152997 t_x_c,CourseSession +0.10575638546049076 t_x_c,User +0.21980738938847105 t_x_c,Course -0.2156600801547263 t_dist_c,User,Therapist -0.00994955436364722 t_dist_c,Clinic,Location -0.715955841790786 t_dist_c,Course,CourseSession -0.2156600801547263 t_dist_c,Clinic,Therapist -0.10990369469423553 t_center_dist_c,Role -0.10990369469423553 t_dist_c,Role,User -0.10990369469423553 t_center_dist_c,CourseSession -0.01989910872729444 t_dist_c,Client,Location -0.586153038369256 t_dist_r,Course,CourseSession -0.071068245454623 t_dist_c,Client,CourseSession -0.07106824545462298 t_dist_r,Client,CourseSession <= -1.6905298358451655
 newcut_50: -2.0 t_zV_r,rt,CertificateTemplate,CourseSession -1.0 t_zV_c,rt,CertificateTemplate,CourseSession -1.0 t_zV_r,lb,CertificateTemplate,CourseSession -1.0 t_zV_c,lb,CertificateTemplate,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Clinic -0.06174687532769513 t_zV_r,rt,Role,User -0.06174687532769514 t_zV_c,rt,Role,User -0.06174687532769513 t_zV_r,lb,Role,User -0.06174687532769514 t_zV_c,lb,Role,User -1.0 t_zV_r,rt,CourseSession,Therapist -1.0 t_zV_c,rt,CourseSession,Therapist -1.0 t_zV_r,lb,CourseSession,Therapist -1.0 t_zV_c,lb,CourseSession,Therapist +0.004749759640591933 t_x_c,Location -0.039803813632977934 t_x_c,Clinic -0.05405309255475373 t_x_c,CourseSession +0.08910714654713973 t_x_c,CertificateTemplate -0.009499519281183868 t_x_r,Instructor -0.020116629066036427 t_x_r,Therapist +0.009499519281183868 t_x_r,Location -0.0284985578435516 t_x_r,User -0.05336494655017996 t_x_r,Course -0.009499519281183868 t_center_dist_r,Clinic -0.009499519281183868 t_dist_r,Role,User -0.0284985578435516 t_dist_c,User,Therapist -0.009499519281183868 t_center_dist_c,Location -0.039803813632977934 t_dist_c,Clinic,Location -0.6586333368287479 t_dist_r,Therapist,CourseSession -0.6280837934497602 t_dist_r,CertificateTemplate,CourseSession -0.009499519281183868 t_center_dist_r,Location -0.009499519281183868 t_dist_r,Client,User -0.0284985578435516 t_dist_r,User,Therapist -0.30994509968411676 t_dist_c,Course,CourseSession -0.009499519281183868 t_center_dist_c,Clinic -0.035054053992386 t_dist_c,Location,CourseSession -0.009499519281183868 t_center_dist_c,Role -0.009499519281183868 t_center_dist_r,Instructor -0.009499519281183868 t_center_dist_c,CertificateTemplate -0.009499519281183868 t_center_dist_c,Course -0.005880654793113822 t_dist_c,Role,User -0.009499519281183868 t_center_dist_c,CourseSession -0.05811470619077189 t_center_dist_r,Therapist -0.009499519281183868 t_dist_c,Client,Location -0.009499519281183868 t_dist_r,Instructor,Location -0.06286446583136383 t_center_dist_r,Course -0.009499519281183868 t_center_dist_c,User -0.7171909399968999 t_dist_c,CertificateTemplate,CourseSession -0.009499519281183868 t_center_dist_r,Client -0.08773085453799218 t_center_dist_r,CertificateTemplate -0.009499519281183868 t_center_dist_r,Role -0.009499519281183868 t_center_dist_c,Client -0.009499519281183868 t_dist_c,Client,Instructor -0.009499519281183868 t_dist_c,Client,User -0.009499519281183868 t_center_dist_c,Instructor -0.30994509968411676 t_dist_r,Course,CourseSession -0.009499519281183868 t_center_dist_r,User -0.009499519281183868 t_center_dist_c,Therapist -0.00784742897141276 t_dist_c,Instructor,Location -0.6586333368287479 t_dist_c,Therapist,CourseSession -0.009499519281183868 t_center_dist_r,CourseSession <= -17.97565433403914
 newcut_51: +1.0 t_zV_c,lb,Course,Role +1.0 t_zV_c,lb,Role,Clinic -0.02439024390243903 t_x_c,Clinic -0.02439024390243903 t_x_c,CourseSession +0.04878048780487806 t_x_c,Course -0.02439024390243903 t_dist_c,Clinic,Location -0.04878048780487806 t_dist_c,Course,CourseSession -0.02439024390243903 t_dist_c,Location,CourseSession <= 1.0609756097560978
 newcut_52: -1.0 t_zV_r,lb,Location,Therapist +1.0 t_zV_c,lb,User,Location -1.0 t_zV_r,rt,User,Therapist +0.15883636363636358 t_x_c,Location -0.24203636363636352 t_x_c,Clinic +0.08319999999999995 t_x_c,User -0.4008727272727271 t_dist_c,User,Therapist -0.24203636363636352 t_dist_c,Clinic,Location -0.08479999999999996 t_dist_r,Client,User -0.005191836734693863 t_dist_r,User,Therapist -0.48407272727272704 t_dist_c,Clinic,Therapist -0.4008727272727271 t_dist_c,Location,Therapist -0.08479999999999996 t_dist_c,Client,Location -0.08479999999999996 t_dist_r,Client,Location -0.08479999999999996 t_dist_c,Client,User <= -1.4857306122448972
 newcut_53: -0.5 t_x_c,Client +0.5 t_x_c,Instructor -1.0 t_dist_c,Client,Instructor <= -0.5
 newcut_54: +0.5 t_x_c,Client -0.5 t_x_c,Instructor -1.0 t_dist_c,Client,Instructor <= 0.0
 newcut_55: -1.632067441029354 t_zV_r,rt,Client,User -1.626742410987379 t_zV_c,rt,Client,User -1.626742410987379 t_zV_c,lb,Client,User -0.7775713152488731 t_zV_r,rt,Client,CourseSession -0.7775713152488731 t_zV_c,rt,Client,CourseSession -0.7775713152488731 t_zV_r,lb,Client,CourseSession -0.7775713152488731 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,Client,Therapist +1.075104597654489 t_zV_c,lb,CertificateTemplate,User -0.6320674410293547 t_zV_c,rt,CertificateTemplate,CourseSession -0.6320674410293547 t_zV_c,lb,CertificateTemplate,CourseSession +0.6320674410293551 t_zV_r,lb,CertificateTemplate,Instructor -0.023824209667391752 t_zV_r,rt,Role,User -0.023824209667391752 t_zV_c,rt,Role,User -0.023824209667391752 t_zV_r,lb,Role,User -0.023824209667391752 t_zV_c,lb,Role,User +1.632067441029354 t_zV_r,lb,User,Therapist +0.6320674410293544 t_zV_r,lb,CourseSession,Clinic -0.5722944637086714 t_zV_r,rt,CourseSession,Location -0.5722944637086715 t_zV_c,rt,CourseSession,Location -0.5722944637086714 t_zV_r,lb,CourseSession,Location -0.5722944637086715 t_zV_c,lb,CourseSession,Location -0.18692841431338153 t_zV_c,rt,CourseSession,Therapist -0.18692841431338153 t_zV_c,lb,CourseSession,Therapist -0.6320674410293551 t_zV_c,rt,CourseSession,Instructor -0.6320674410293551 t_zV_c,lb,CourseSession,Instructor -0.048222042423658525 t_x_c,Therapist +0.048236819935866204 t_x_c,Location -0.05415289470627252 t_x_c,Clinic -0.10305228477702404 t_x_c,CourseSession +0.07854262077562017 t_x_c,User +0.10256387956099879 t_x_c,CertificateTemplate -0.01700349170263185 t_x_c,Client -0.053588314865002734 t_x_r,Instructor -0.004749958972888301 t_x_r,Therapist +0.053588314865002734 t_x_r,Location -0.019616604611017455 t_x_r,Clinic +0.05341414994614268 t_x_r,CourseSession -0.284457718273116 t_x_r,User -0.021591856909340238 t_x_r,Course +0.23052107357039314 t_x_r,Client -0.00691260666289823 t_x_c,Instructor -0.03759548617859175 t_center_dist_r,Clinic -0.017978881567574285 t_dist_r,Role,User -0.017804716648714223 t_dist_r,Clinic,Location -0.22042124925694123 t_dist_c,User,Therapist -0.01797888156757428 t_center_dist_c,Location -0.07195761135498674 t_dist_c,Clinic,Location -0.06358827374540794 t_dist_r,Location,CourseSession -0.019676675190882266 t_dist_r,Therapist,CourseSession -0.017978881567574285 t_center_dist_r,Location -0.55935294744283 t_dist_r,Client,User -0.10354937722923747 t_dist_r,User,Therapist -0.13039003178969005 t_dist_c,Course,CourseSession -0.01797888156757428 t_center_dist_c,Clinic -0.12611050606125926 t_dist_c,Clinic,Therapist -0.06358827374540794 t_dist_c,Location,CourseSession -0.1023771207301068 t_dist_r,Client,Instructor -0.01797888156757428 t_center_dist_c,Role -0.017978881567574285 t_center_dist_r,Instructor -0.017978881567574285 t_center_dist_c,CertificateTemplate -0.017978881567574285 t_center_dist_c,Course -0.017912533027056164 t_dist_c,Role,User -0.017978881567574285 t_center_dist_c,CourseSession -0.050897765518320315 t_center_dist_r,Therapist -0.1469015062639243 t_dist_c,Client,Location -0.05341414994614268 t_dist_r,Client,Location -0.03927469632318103 t_dist_r,Instructor,Location -0.03957073847691452 t_center_dist_r,Course -0.01797888156757428 t_center_dist_c,User -0.09366152123664168 t_dist_c,CertificateTemplate,CourseSession -0.017804716648714223 t_dist_c,CourseSession,Instructor -0.008902358324357111 t_dist_c,Client,CertificateTemplate -0.017978881567574285 t_center_dist_r,Client -0.058820555040257995 t_center_dist_r,CertificateTemplate -0.017978881567574285 t_center_dist_r,Role -0.01797888156757428 t_center_dist_c,Client -0.11312915533761768 t_dist_c,Client,Instructor -0.8512110439019356 t_dist_c,Client,User -0.017978881567574285 t_center_dist_c,Instructor -0.13039003178969008 t_dist_r,Course,CourseSession -0.33450543298800606 t_dist_c,Client,CourseSession -0.017978881567574285 t_center_dist_r,User -0.01797888156757428 t_center_dist_c,Therapist -0.04444314513893522 t_dist_c,Instructor,Location -0.017804716648714223 t_dist_r,CourseSession,Instructor -0.33450543298800606 t_dist_r,Client,CourseSession -0.07024207047323061 t_dist_c,Therapist,CourseSession -0.017978881567574285 t_center_dist_r,CourseSession <= -11.17751928066997
 newcut_56: -0.06841017779481111 t_zV_c,lb,Location,Instructor -0.0684101777948111 t_zV_r,lb,Location,Instructor -0.06841017779481111 t_zV_c,rt,Location,Instructor -0.0684101777948111 t_zV_r,rt,Location,Instructor +1.0 t_zV_r,lb,CourseSession,Clinic +1.4978228401390221 t_zV_r,rt,CourseSession,Therapist +1.4978228401390221 t_zV_r,lb,CourseSession,Therapist -0.014402142693644444 t_x_c,CourseSession +0.007201071346822222 t_x_c,CertificateTemplate +0.007201071346822222 t_x_c,Client -0.007201071346822222 t_x_r,Instructor -0.01524932755797647 t_x_r,Therapist +0.007201071346822222 t_x_r,Location -0.036852541598443124 t_x_r,Clinic +0.06650401185006405 t_x_r,CourseSession -0.021603214040466667 t_x_r,User -0.04045307727185424 t_x_r,Course -0.036852541598443124 t_center_dist_r,Clinic -0.007201071346822222 t_dist_r,Role,User -0.021603214040466664 t_dist_c,User,Therapist -0.007201071346822222 t_center_dist_c,Location -0.15766556211989707 t_dist_r,Therapist,CourseSession -0.007201071346822222 t_center_dist_r,Location -0.007201071346822222 t_dist_r,Client,User -0.021603214040466664 t_dist_r,User,Therapist -0.23495260237474855 t_dist_c,Course,CourseSession -0.007201071346822222 t_center_dist_c,Clinic -0.007201071346822222 t_center_dist_c,Role -0.007201071346822222 t_center_dist_r,Instructor -0.007201071346822222 t_center_dist_c,CertificateTemplate -0.007201071346822222 t_center_dist_c,Course -0.004457806071842327 t_dist_c,Role,User -0.007201071346822222 t_center_dist_c,CourseSession -0.04405361294526536 t_center_dist_r,Therapist -0.007201071346822222 t_dist_r,Instructor,Location -0.04765414861867648 t_center_dist_r,Course -0.007201071346822222 t_center_dist_c,User -0.007201071346822222 t_dist_c,CertificateTemplate,CourseSession -0.007201071346822222 t_center_dist_r,Client -0.007201071346822222 t_center_dist_r,CertificateTemplate -0.007201071346822222 t_center_dist_r,Role -0.007201071346822222 t_center_dist_c,Client -0.007201071346822222 t_dist_c,Client,Instructor -0.007201071346822222 t_dist_c,Client,User -0.007201071346822222 t_center_dist_c,Instructor -0.23495260237474858 t_dist_r,Course,CourseSession -0.025718111952936502 t_dist_c,Client,CourseSession -0.007201071346822222 t_center_dist_r,User -0.007201071346822222 t_center_dist_c,Therapist -0.0059487111125922695 t_dist_c,Instructor,Location -0.025718111952936502 t_dist_r,Client,CourseSession -0.15766556211989707 t_dist_c,Therapist,CourseSession -0.007201071346822222 t_center_dist_r,CourseSession <= -2.856152658152136
 newcut_57: +0.30728148507264313 t_zV_c,lb,Course,Instructor -1.0 t_zV_r,rt,User,Instructor -0.6927185149273569 t_zV_c,rt,User,Instructor -0.6927185149273569 t_zV_r,lb,User,Instructor -0.6927185149273569 t_zV_c,lb,User,Instructor +0.013218804247532969 t_x_c,Location -0.013218804247532969 t_x_c,CourseSession +0.026437608495065937 t_x_c,Course -0.028056237586600585 t_x_r,Instructor +0.028056237586600585 t_x_r,Location -0.026437608495065937 t_x_c,Instructor -0.4582518805811428 t_dist_r,User,Instructor -0.028056237586600585 t_dist_r,Client,User -0.026437608495065937 t_dist_c,Course,CourseSession -0.013218804247532969 t_dist_c,Location,CourseSession -0.054493846081666526 t_dist_c,Client,Location -0.028056237586600585 t_dist_r,Client,Location -0.028056237586600585 t_dist_r,Instructor,Location -0.4582518805811428 t_dist_c,User,Instructor -0.026437608495065937 t_dist_c,Client,Instructor -0.028056237586600585 t_dist_c,Client,User -0.023176891919365703 t_dist_c,Instructor,Location <= -5.016110051139298
 newcut_58: +0.47121559119883183 t_zV_r,rt,Therapist,Instructor -0.09705936191936736 t_zV_c,lb,Location,Instructor -0.09705936191936737 t_zV_r,lb,Location,Instructor -0.09705936191936736 t_zV_c,rt,Location,Instructor -0.09705936191936737 t_zV_r,rt,Location,Instructor -0.13281807420545008 t_zV_r,rt,Role,User -0.13281807420545008 t_zV_c,rt,Role,User -0.13281807420545008 t_zV_r,lb,Role,User -0.13281807420545008 t_zV_c,lb,Role,User +0.4712155911988316 t_zV_r,lb,Role,Instructor +0.010216774938880775 t_x_c,CertificateTemplate -0.012574492232468643 t_x_r,Instructor -0.00903791629208684 t_x_r,Therapist +0.010216774938880775 t_x_r,Location +0.010216774938880775 t_x_r,CourseSession -0.02043354987776155 t_x_r,User +0.020433549877761553 t_x_r,Role -0.007859057645292907 t_x_r,Course -0.010216774938880775 t_x_c,Instructor -0.010216774938880775 t_center_dist_r,Clinic -0.02043354987776155 t_dist_r,Role,User -0.010216774938880775 t_center_dist_c,Location -0.010216774938880775 t_center_dist_r,Location -0.010216774938880775 t_dist_r,Client,User -0.01807583258417368 t_dist_c,Course,CourseSession -0.010216774938880775 t_center_dist_c,Clinic -0.010216774938880775 t_center_dist_c,Role -0.010216774938880775 t_center_dist_r,Instructor -0.010216774938880775 t_center_dist_c,CertificateTemplate -0.010216774938880775 t_center_dist_c,Course -0.012649340400519056 t_dist_c,Role,User -0.010216774938880775 t_center_dist_c,CourseSession -0.019254691230967617 t_center_dist_r,Therapist -0.010216774938880775 t_dist_r,Instructor,Location -0.010216774938880775 t_center_dist_r,Course -0.010216774938880775 t_center_dist_c,User -0.010216774938880775 t_dist_c,CertificateTemplate,CourseSession -0.010216774938880775 t_dist_c,CourseSession,Instructor -0.010216774938880775 t_center_dist_r,Client -0.010216774938880775 t_center_dist_r,CertificateTemplate -0.010216774938880775 t_center_dist_r,Role -0.010216774938880775 t_center_dist_c,Client -0.010216774938880775 t_dist_c,Client,User -0.010216774938880775 t_center_dist_c,Instructor -0.007859057645292907 t_dist_r,Course,CourseSession -0.010216774938880775 t_center_dist_r,User -0.010216774938880775 t_center_dist_c,Therapist -0.008439944514727597 t_dist_c,Instructor,Location -0.002357717293587868 t_dist_r,CourseSession,Instructor -0.010216774938880775 t_center_dist_r,CourseSession <= -0.30637307743937914
 newcut_59: -1.0 t_zV_r,rt,Clinic,Therapist +1.0 t_zV_c,lb,Client,Therapist -0.048685491723466395 t_x_c,Therapist -0.1172877755156236 t_x_c,Clinic +0.2832610427547136 t_x_c,User -0.1172877755156236 t_x_c,Client -0.025803310613437193 t_x_r,Therapist +0.025803310613437193 t_x_r,Clinic -0.025803310613437193 t_center_dist_r,Clinic -0.2832610427547136 t_dist_c,User,Therapist -0.1172877755156236 t_dist_c,Clinic,Location -0.2345755510312472 t_dist_c,Clinic,Therapist -0.025803310613437193 t_center_dist_r,Therapist -0.1172877755156236 t_dist_c,Client,Location -0.2832610427547136 t_dist_c,Client,User <= 0.025803310613437193
 newcut_60: +1.0 t_zV_c,lb,Client,Clinic +1.0 t_zV_c,rt,CourseSession,Clinic +0.021276595744680847 t_x_c,Client -0.021276595744680847 t_x_c,Instructor -0.042553191489361694 t_dist_c,CourseSession,Instructor -0.042553191489361694 t_dist_c,Client,Instructor <= 1.0000000000000002
 newcut_61: -0.09822323652763235 t_zV_c,lb,Location,Instructor -0.09822323652763235 t_zV_r,lb,Location,Instructor -0.09822323652763235 t_zV_c,rt,Location,Instructor -0.09822323652763235 t_zV_r,rt,Location,Instructor -1.0 t_zV_r,rt,Role,User -0.13441074472202308 t_zV_c,rt,Role,User -0.13441074472202308 t_zV_c,lb,Role,User +0.48885747999779494 t_zV_r,lb,Role,Instructor -1.0 t_zV_r,rt,User,Instructor +0.010339288055540246 t_x_c,CertificateTemplate -0.13534656349493332 t_x_r,Instructor +0.10432869932831258 t_x_r,Location +0.031017864166620723 t_x_r,CourseSession -0.10650221389765245 t_x_r,User +0.10650221389765244 t_x_r,Role -0.005169644027770115 t_x_r,Course -0.010339288055540246 t_x_c,Instructor -0.010339288055540246 t_center_dist_r,Clinic -0.10650221389765244 t_dist_r,Role,User -0.010339288055540246 t_center_dist_c,Location -0.010339288055540246 t_center_dist_r,Location -0.10432869932831258 t_dist_r,Client,User -0.020678576111080472 t_dist_c,Course,CourseSession -0.010339288055540246 t_center_dist_c,Clinic -0.010339288055540246 t_center_dist_c,Role -0.010339288055540246 t_center_dist_r,Instructor -0.010339288055540246 t_center_dist_c,CertificateTemplate -0.010339288055540246 t_center_dist_c,Course -0.23549714106677355 t_dist_c,Role,User -0.010339288055540246 t_center_dist_c,CourseSession -0.010339288055540246 t_center_dist_r,Therapist -0.10432869932831258 t_dist_c,Client,Location -0.10432869932831258 t_dist_r,Client,Location -0.10432869932831258 t_dist_r,Instructor,Location -0.1294658678258952 t_dist_c,User,Instructor -0.01550893208331036 t_center_dist_r,Course -0.010339288055540246 t_center_dist_c,User -0.010339288055540246 t_dist_c,CertificateTemplate,CourseSession -0.03101786416662072 t_dist_c,CourseSession,Instructor -0.010339288055540246 t_center_dist_r,Client -0.010339288055540246 t_center_dist_r,CertificateTemplate -0.010339288055540246 t_center_dist_r,Role -0.010339288055540246 t_center_dist_c,Client -0.10432869932831258 t_dist_c,Client,User -0.010339288055540246 t_center_dist_c,Instructor -0.020678576111080475 t_dist_r,Course,CourseSession -0.010339288055540246 t_center_dist_r,User -0.010339288055540246 t_center_dist_c,Therapist -0.10432869932831258 t_dist_c,Instructor,Location -0.03101786416662072 t_dist_r,CourseSession,Instructor -0.010339288055540246 t_center_dist_r,CourseSession <= -3.3143779354395972
 newcut_62: -0.21777452919425555 t_zV_c,lb,Location,Instructor -0.21777452919425555 t_zV_r,lb,Location,Instructor -0.21777452919425555 t_zV_c,rt,Location,Instructor -0.21777452919425555 t_zV_r,rt,Location,Instructor -0.8646659214033889 t_zV_r,rt,Client,Instructor -0.8646659214033885 t_zV_c,rt,Client,Instructor -0.8646659214033889 t_zV_r,lb,Client,Instructor -0.8646659214033885 t_zV_c,lb,Client,Instructor -0.43233296070169447 t_zV_r,rt,CourseSession,Location -0.43233296070169447 t_zV_c,rt,CourseSession,Location -0.43233296070169447 t_zV_c,lb,CourseSession,Location +1.0 t_zV_r,lb,CourseSession,Instructor +0.010086851835927425 t_x_c,Location -0.020173703671854847 t_x_c,Clinic +0.040347407343709694 t_x_c,User +0.040347407343709694 t_x_c,CertificateTemplate -0.14912718204488778 t_x_r,Instructor +0.12104222203112908 t_x_r,Location +0.14912718204488778 t_x_r,CourseSession -0.08069481468741939 t_x_r,User -0.020173703671854844 t_x_r,Course -0.040347407343709694 t_x_r,Client -0.07060796285149197 t_x_c,Instructor -0.040347407343709694 t_center_dist_r,Clinic -0.040347407343709694 t_dist_r,Role,User -0.040347407343709694 t_dist_r,Clinic,Location -0.040347407343709694 t_dist_c,User,Therapist -0.040347407343709694 t_center_dist_c,Location -0.06052111101556454 t_dist_c,Clinic,Location -0.14409788337039175 t_dist_r,Location,CourseSession -0.040347407343709694 t_center_dist_r,Location -0.08069481468741939 t_dist_r,Client,User -0.08069481468741939 t_dist_c,Course,CourseSession -0.040347407343709694 t_center_dist_c,Clinic -0.040347407343709694 t_dist_c,Clinic,Therapist -0.14409788337039175 t_dist_c,Location,CourseSession -0.23199759222633073 t_dist_r,Client,Instructor -0.040347407343709694 t_center_dist_c,Role -0.040347407343709694 t_center_dist_r,Instructor -0.040347407343709694 t_center_dist_c,CertificateTemplate -0.040347407343709694 t_center_dist_c,Course -0.040347407343709694 t_dist_c,Role,User -0.0403474073437097 t_center_dist_c,CourseSession -0.040347407343709694 t_center_dist_r,Therapist -0.15130277753891136 t_dist_c,Client,Location -0.12104222203112909 t_dist_r,Client,Location -0.1087797747011781 t_dist_r,Instructor,Location -0.060521111015564545 t_center_dist_r,Course -0.040347407343709694 t_center_dist_c,User -0.020173703671854844 t_dist_c,CertificateTemplate,CourseSession -0.06843236735746841 t_dist_c,CourseSession,Instructor -0.020173703671854847 t_dist_c,Client,CertificateTemplate -0.040347407343709694 t_center_dist_r,Client -0.040347407343709694 t_center_dist_r,CertificateTemplate -0.040347407343709694 t_center_dist_r,Role -0.040347407343709694 t_center_dist_c,Client -0.2521712958981856 t_dist_c,Client,Instructor -0.12104222203112909 t_dist_c,Client,User -0.040347407343709694 t_center_dist_c,Instructor -0.08069481468741939 t_dist_r,Course,CourseSession -0.040347407343709694 t_center_dist_r,User -0.040347407343709694 t_center_dist_c,Therapist -0.11705219709347323 t_dist_c,Instructor,Location -0.06843236735746841 t_dist_r,CourseSession,Instructor -0.040347407343709694 t_center_dist_r,CourseSession <= -8.18349630848986
 newcut_63: -1.0 t_zV_r,rt,Clinic,Therapist +1.0 t_zV_c,lb,CertificateTemplate,Therapist -0.047125353440150786 t_x_c,Therapist -0.11567132208037012 t_x_c,Clinic +0.278467997600891 t_x_c,User +0.047125353440150786 t_x_c,CertificateTemplate -0.1392339988004455 t_x_c,Client -0.025447690857681428 t_x_r,Therapist +0.025447690857681428 t_x_r,Clinic -0.023562676720075393 t_x_c,Instructor -0.025447690857681428 t_center_dist_r,Clinic -0.278467997600891 t_dist_c,User,Therapist -0.11567132208037012 t_dist_c,Clinic,Location -0.23134264416074024 t_dist_c,Clinic,Therapist -0.025447690857681428 t_center_dist_r,Therapist -0.11567132208037012 t_dist_c,Client,Location -0.047125353440150786 t_dist_c,CertificateTemplate,CourseSession -0.047125353440150786 t_dist_c,CourseSession,Instructor -0.047125353440150786 t_dist_c,Client,Instructor -0.278467997600891 t_dist_c,Client,User <= 0.0018850141376060335
 newcut_64: -0.19291759793213833 t_zV_c,lb,Location,Instructor -0.19291759793213836 t_zV_r,lb,Location,Instructor -0.19291759793213833 t_zV_c,rt,Location,Instructor -0.19291759793213836 t_zV_r,rt,Location,Instructor -1.0 t_zV_r,rt,Client,CertificateTemplate -1.0 t_zV_r,lb,Client,CertificateTemplate -0.18725823785475032 t_zV_r,rt,Client,CourseSession -0.18725823785475035 t_zV_c,rt,Client,CourseSession -0.18725823785475032 t_zV_r,lb,Client,CourseSession -0.18725823785475035 t_zV_c,lb,Client,CourseSession +0.28051344339252826 t_zV_r,lb,CertificateTemplate,Instructor -0.037867776988405065 t_zV_r,rt,Role,User -0.03786777698840507 t_zV_c,rt,Role,User -0.037867776988405065 t_zV_r,lb,Role,User -0.03786777698840507 t_zV_c,lb,Role,User +0.14714336201208827 t_zV_r,rt,CourseSession,Therapist +0.14714336201208827 t_zV_r,lb,CourseSession,Therapist -0.011651623688740021 t_x_c,CourseSession +0.005825811844370011 t_x_c,CertificateTemplate +0.005825811844370011 t_x_c,Client -0.026132927416174047 t_x_r,Instructor -0.0014980659028380025 t_x_r,Therapist +0.026132927416174047 t_x_r,Location +0.00582581184437001 t_x_r,CourseSession -0.020307115571804035 t_x_r,User -0.004327745941532008 t_x_r,Course -0.005825811844370012 t_x_r,Client -0.005825811844370011 t_center_dist_r,Clinic -0.005825811844370011 t_dist_r,Role,User -0.008655491883064017 t_dist_c,User,Therapist -0.005825811844370011 t_center_dist_c,Location -0.01548877494864087 t_dist_r,Therapist,CourseSession -0.005825811844370011 t_center_dist_r,Location -0.020307115571804035 t_dist_r,Client,User -0.008655491883064017 t_dist_r,User,Therapist -0.028907123532540716 t_dist_c,Course,CourseSession -0.005825811844370011 t_center_dist_c,Clinic -0.005825811844370011 t_center_dist_c,Role -0.005825811844370011 t_center_dist_r,Instructor -0.005825811844370011 t_center_dist_c,CertificateTemplate -0.005825811844370011 t_center_dist_c,Course -0.003606454951276673 t_dist_c,Role,User -0.005825811844370011 t_center_dist_c,CourseSession -0.010153557785902018 t_center_dist_r,Therapist -0.026132927416174047 t_dist_c,Client,Location -0.026132927416174047 t_dist_r,Client,Location -0.020307115571804035 t_dist_r,Instructor,Location -0.010153557785902018 t_center_dist_r,Course -0.005825811844370011 t_center_dist_c,User -0.005825811844370011 t_dist_c,CertificateTemplate,CourseSession -0.11134899507761116 t_dist_c,Client,CertificateTemplate -0.005825811844370011 t_center_dist_r,Client -0.03195873926054406 t_center_dist_r,CertificateTemplate -0.005825811844370011 t_center_dist_r,Role -0.005825811844370011 t_center_dist_c,Client -0.005825811844370011 t_dist_c,Client,Instructor -0.020307115571804035 t_dist_c,Client,User -0.005825811844370011 t_center_dist_c,Instructor -0.023081311688170708 t_dist_r,Course,CourseSession -0.020806470872750036 t_dist_c,Client,CourseSession -0.005825811844370011 t_center_dist_r,User -0.005825811844370011 t_center_dist_c,Therapist -0.016775443298446813 t_dist_c,Instructor,Location -0.02080647087275004 t_dist_r,Client,CourseSession -0.01548877494864087 t_dist_c,Therapist,CourseSession -0.005825811844370011 t_center_dist_r,CourseSession <= -2.6725447459740987
 newcut_65: +0.709671205955362 t_zV_r,lb,Role,User +0.290328794044638 t_zV_c,lb,User,Location +1.0 t_zV_r,lb,User,Instructor -0.009433962264150945 t_x_c,Location -0.009433962264150945 t_x_c,Clinic +0.01886792452830189 t_x_c,User -0.07007299270072989 t_x_r,Instructor +0.07007299270072989 t_x_r,Location -0.07153284671532843 t_x_r,User +0.07153284671532843 t_x_r,Role -0.07153284671532843 t_dist_r,Role,User -0.01886792452830189 t_dist_c,User,Therapist -0.009433962264150945 t_dist_c,Clinic,Location -0.07007299270072989 t_dist_r,Client,User -0.01886792452830189 t_dist_c,Clinic,Therapist -0.07153284671532843 t_dist_c,Role,User -0.07007299270072989 t_dist_c,Client,Location -0.07007299270072989 t_dist_r,Client,Location -0.07007299270072989 t_dist_r,Instructor,Location -0.07007299270072989 t_dist_c,Client,User -0.07007299270072989 t_dist_c,Instructor,Location <= -0.12043795620437941
 newcut_66: -0.17839001993663633 t_zV_c,lb,Location,Instructor -0.17839001993663636 t_zV_r,lb,Location,Instructor -0.17839001993663633 t_zV_c,rt,Location,Instructor -0.17839001993663636 t_zV_r,rt,Location,Instructor +1.0 t_zV_c,lb,User,Location -1.0 t_zV_r,rt,CourseSession,Location -2.032630338922817 t_zV_r,rt,CourseSession,Therapist -2.032630338922817 t_zV_c,rt,CourseSession,Therapist -2.032630338922817 t_zV_r,lb,CourseSession,Therapist -2.032630338922817 t_zV_c,lb,CourseSession,Therapist -0.027277688672229884 t_x_c,Location -0.027277688672229912 t_x_c,Clinic -0.01877789683543543 t_x_c,CourseSession +0.054555377344459824 t_x_c,User +0.018777896835435404 t_x_c,CertificateTemplate -0.018777896835435404 t_x_r,Instructor -0.009388948417717698 t_x_r,Therapist +0.018777896835435404 t_x_r,Location +0.05669480390698767 t_x_r,CourseSession -0.03791690707155226 t_x_r,User -0.018777896835435407 t_x_r,Course -0.018777896835435404 t_x_r,Client -0.018777896835435404 t_center_dist_r,Clinic -0.018777896835435404 t_dist_r,Role,User -0.054555377344459824 t_dist_c,User,Therapist -0.018777896835435404 t_center_dist_c,Location -0.027277688672229912 t_dist_c,Clinic,Location -0.3192242462024018 t_dist_r,Therapist,CourseSession -0.048822531772132055 t_dist_r,CertificateTemplate,CourseSession -0.018777896835435404 t_center_dist_r,Location -0.03791690707155226 t_dist_r,Client,User -0.00036111340068145295 t_dist_r,User,Therapist -0.15022317468348323 t_dist_c,Course,CourseSession -0.018777896835435404 t_center_dist_c,Clinic -0.054555377344459824 t_dist_c,Clinic,Therapist -0.23131479994050966 t_dist_c,Location,CourseSession -0.018777896835435404 t_center_dist_c,Role -0.018777896835435404 t_center_dist_r,Instructor -0.018777896835435404 t_center_dist_c,CertificateTemplate -0.018777896835435404 t_center_dist_c,Course -0.011624412326698107 t_dist_c,Role,User -0.018777896835435404 t_center_dist_c,CourseSession -0.02816684525315311 t_center_dist_r,Therapist -0.07547270074242307 t_dist_c,Client,Location -0.05669480390698766 t_dist_r,Client,Location -0.018777896835435404 t_dist_r,Instructor,Location -0.03755579367087081 t_center_dist_r,Course -0.018777896835435404 t_center_dist_c,User -0.048822531772132055 t_dist_c,CertificateTemplate,CourseSession -0.018777896835435404 t_center_dist_r,Client -0.018777896835435404 t_center_dist_r,CertificateTemplate -0.018777896835435404 t_center_dist_r,Role -0.018777896835435404 t_center_dist_c,Client -0.018777896835435404 t_dist_c,Client,Instructor -0.03791690707155226 t_dist_c,Client,User -0.018777896835435404 t_center_dist_c,Instructor -0.15022317468348326 t_dist_r,Course,CourseSession -0.018777896835435404 t_center_dist_r,User -0.018777896835435404 t_center_dist_c,Therapist -0.015512175646664031 t_dist_c,Instructor,Location -0.3192242462024018 t_dist_c,Therapist,CourseSession -0.018777896835435404 t_center_dist_r,CourseSession <= -8.093628159052788
 newcut_67: +0.42862029417090897 t_zV_r,lb,CertificateTemplate,User -0.06317028511640074 t_zV_r,rt,Role,User -0.06317028511640076 t_zV_c,rt,Role,User -0.06317028511640074 t_zV_r,lb,Role,User -0.06317028511640076 t_zV_c,lb,Role,User +0.009718505402523189 t_x_c,CertificateTemplate -0.009718505402523189 t_x_r,Instructor +0.009718505402523189 t_x_r,Location +0.009718505402523189 t_x_r,CourseSession -0.029155516207569572 t_x_r,User -0.009718505402523189 t_x_r,Course -0.009718505402523189 t_x_c,Instructor -0.009718505402523189 t_center_dist_r,Clinic -0.009718505402523192 t_dist_r,Role,User -0.009718505402523189 t_center_dist_c,Location -0.009718505402523189 t_center_dist_r,Location -0.009718505402523189 t_dist_r,Client,User -0.019437010805046378 t_dist_c,Course,CourseSession -0.009718505402523189 t_center_dist_c,Clinic -0.009718505402523189 t_center_dist_c,Role -0.009718505402523189 t_center_dist_r,Instructor -0.009718505402523189 t_center_dist_c,CertificateTemplate -0.009718505402523189 t_center_dist_c,Course -0.006016217630133405 t_dist_c,Role,User -0.009718505402523189 t_center_dist_c,CourseSession -0.009718505402523189 t_center_dist_r,Therapist -0.009718505402523189 t_dist_r,Instructor,Location -0.009718505402523189 t_center_dist_r,Course -0.009718505402523189 t_center_dist_c,User -0.009718505402523189 t_dist_c,CertificateTemplate,CourseSession -0.009718505402523189 t_dist_c,CourseSession,Instructor -0.009718505402523189 t_center_dist_r,Client -0.03887402161009276 t_center_dist_r,CertificateTemplate -0.009718505402523189 t_center_dist_r,Role -0.009718505402523189 t_center_dist_c,Client -0.009718505402523189 t_dist_c,Client,User -0.009718505402523189 t_center_dist_c,Instructor -0.009718505402523189 t_dist_r,Course,CourseSession -0.009718505402523189 t_center_dist_r,User -0.009718505402523189 t_center_dist_c,Therapist -0.00802833054991046 t_dist_c,Instructor,Location -0.009718505402523189 t_center_dist_r,CourseSession <= -0.8242218153282762
 newcut_68: +1.0 t_zV_c,lb,Course,User -0.9392999556933981 t_zV_r,rt,Course,CourseSession -0.939299955693398 t_zV_c,rt,Course,CourseSession -0.9392999556933981 t_zV_r,lb,Course,CourseSession -0.939299955693398 t_zV_c,lb,Course,CourseSession +1.0 t_zV_c,lb,User,Location -0.0389265143363507 t_zV_r,rt,CourseSession,Location -0.0389265143363507 t_zV_c,rt,CourseSession,Location -0.0389265143363507 t_zV_r,lb,CourseSession,Location -0.0389265143363507 t_zV_c,lb,CourseSession,Location -0.0465219317678334 t_x_c,CourseSession -0.0004430660168365107 t_x_c,User +0.04696499778466991 t_x_c,Course -0.16456737768213173 t_dist_r,Location,CourseSession -0.12523999409245307 t_dist_c,Course,CourseSession -0.16456737768213173 t_dist_c,Location,CourseSession -0.0004430660168365107 t_center_dist_c,Role -0.0004430660168365107 t_dist_c,Role,User -0.0004430660168365107 t_center_dist_c,CourseSession -0.12523999409245307 t_dist_r,Course,CourseSession <= -2.305367428318247
 newcut_69: -0.11552862726704831 t_zV_c,lb,Location,Instructor -0.1155286272670483 t_zV_r,lb,Location,Instructor -0.11552862726704831 t_zV_c,rt,Location,Instructor -0.1155286272670483 t_zV_r,rt,Location,Instructor +1.751170771205784 t_zV_c,lb,Client,Clinic -1.0 t_zV_r,rt,Client,Instructor -0.1580918057338555 t_zV_r,rt,Role,User -0.1580918057338555 t_zV_c,rt,Role,User -0.1580918057338555 t_zV_r,lb,Role,User -0.1580918057338555 t_zV_c,lb,Role,User +0.751170771205784 t_zV_r,lb,Role,Instructor +0.025247822834538493 t_x_c,Location -0.04176830180817796 t_x_c,Clinic +0.0121609081333735 t_x_c,CertificateTemplate +0.016520478973639445 t_x_c,Client -0.12329527325633352 t_x_r,Instructor +0.08681254885621302 t_x_r,Location +0.03648272440012049 t_x_r,CourseSession -0.024321816266747 t_x_r,User +0.024321816266747003 t_x_r,Role -0.006080454066686745 t_x_r,Course -0.012160908133373472 t_x_c,Instructor -0.0121609081333735 t_center_dist_r,Clinic -0.024321816266747 t_dist_r,Role,User -0.0121609081333735 t_center_dist_c,Location -0.04176830180817796 t_dist_c,Clinic,Location -0.0121609081333735 t_center_dist_r,Location -0.0121609081333735 t_dist_r,Client,User -0.024321816266746992 t_dist_c,Course,CourseSession -0.0121609081333735 t_center_dist_c,Clinic -0.0121609081333735 t_center_dist_c,Role -0.0121609081333735 t_center_dist_r,Instructor -0.0121609081333735 t_center_dist_c,CertificateTemplate -0.0121609081333735 t_center_dist_c,Course -0.01505636245084338 t_dist_c,Role,User -0.012160908133373487 t_center_dist_c,CourseSession -0.0121609081333735 t_center_dist_r,Therapist -0.10333302782985249 t_dist_c,Client,Location -0.08681254885621302 t_dist_r,Client,Location -0.08681254885621302 t_dist_r,Instructor,Location -0.018241362200060247 t_center_dist_r,Course -0.0121609081333735 t_center_dist_c,User -0.0121609081333735 t_dist_c,CertificateTemplate,CourseSession -0.03648272440012049 t_dist_c,CourseSession,Instructor -0.0121609081333735 t_center_dist_r,Client -0.0121609081333735 t_center_dist_r,CertificateTemplate -0.0121609081333735 t_center_dist_r,Role -0.0121609081333735 t_center_dist_c,Client -0.38499478188407515 t_dist_c,Client,Instructor -0.0121609081333735 t_dist_c,Client,User -0.0121609081333735 t_center_dist_c,Instructor -0.024321816266746992 t_dist_r,Course,CourseSession -0.0121609081333735 t_center_dist_r,User -0.0121609081333735 t_center_dist_c,Therapist -0.0717147142725238 t_dist_c,Instructor,Location -0.03648272440012049 t_dist_r,CourseSession,Instructor -0.0121609081333735 t_center_dist_r,CourseSession <= -1.0333963534370239
 newcut_70: -0.24237382140876323 t_zV_c,lb,Location,Instructor -1.0 t_zV_r,lb,Location,Instructor -0.24237382140876323 t_zV_c,rt,Location,Instructor -1.0 t_zV_r,rt,Location,Instructor +0.3011647254575709 t_zV_r,lb,Client,Instructor -0.02551303383250139 t_x_r,Instructor +0.02551303383250139 t_x_r,Location -0.02551303383250139 t_dist_c,Client,Location -0.02551303383250139 t_dist_r,Client,Location -0.02551303383250139 t_dist_r,Instructor,Location -0.09206877426511371 t_dist_c,Instructor,Location <= -1.2168607875762616
 newcut_71: -0.08189896786365215 t_zV_c,lb,Location,Instructor -0.08189896786365215 t_zV_r,lb,Location,Instructor -0.08189896786365215 t_zV_c,rt,Location,Instructor -0.08189896786365215 t_zV_r,rt,Location,Instructor +1.3534882057466726 t_zV_c,lb,Course,Role +1.0 t_zV_c,lb,Role,Therapist -0.03879424793541397 t_zV_r,rt,CourseSession,Therapist -0.3922824536820868 t_zV_c,rt,CourseSession,Therapist -0.3922824536820868 t_zV_r,lb,CourseSession,Therapist -0.3922824536820868 t_zV_c,lb,CourseSession,Therapist -0.0330119074572359 t_x_c,Therapist +0.004310471992823797 t_x_c,Location -0.01650595372861795 t_x_c,Clinic -0.062449277164325245 t_x_c,CourseSession +0.0330119074572359 t_x_c,User +0.0660238149144718 t_x_c,Course +0.008620943985647595 t_x_c,CertificateTemplate -0.008620943985647595 t_x_r,Instructor +0.021552359964118993 t_x_r,Therapist +0.008620943985647595 t_x_r,Location -0.025862831956942785 t_x_r,User -0.008620943985647597 t_x_r,Course -0.008620943985647595 t_center_dist_r,Clinic -0.008620943985647595 t_dist_r,Role,User -0.0660238149144718 t_dist_c,User,Therapist -0.008620943985647595 t_center_dist_c,Location -0.01650595372861795 t_dist_c,Clinic,Location -0.14655604775600914 t_dist_r,Therapist,CourseSession -0.022414454362683748 t_dist_r,CertificateTemplate,CourseSession -0.008620943985647595 t_center_dist_r,Location -0.008620943985647595 t_dist_r,Client,User -0.025862831956942788 t_dist_r,User,Therapist -0.1988168140404859 t_dist_c,Course,CourseSession -0.008620943985647595 t_center_dist_c,Clinic -0.0330119074572359 t_dist_c,Clinic,Therapist -0.02081642572144175 t_dist_c,Location,CourseSession -0.0416328514428835 t_center_dist_c,Role -0.008620943985647595 t_center_dist_r,Instructor -0.008620943985647595 t_center_dist_c,CertificateTemplate -0.008620943985647595 t_center_dist_c,Course -0.03834868230549394 t_dist_c,Role,User -0.0416328514428835 t_center_dist_c,CourseSession -0.012931415978471394 t_center_dist_r,Therapist -0.008620943985647595 t_dist_c,Client,Location -0.008620943985647595 t_dist_r,Instructor,Location -0.01724188797129519 t_center_dist_r,Course -0.008620943985647595 t_center_dist_c,User -0.022414454362683748 t_dist_c,CertificateTemplate,CourseSession -0.008620943985647595 t_center_dist_r,Client -0.008620943985647595 t_center_dist_r,CertificateTemplate -0.008620943985647595 t_center_dist_r,Role -0.008620943985647595 t_center_dist_c,Client -0.008620943985647595 t_dist_c,Client,Instructor -0.008620943985647595 t_dist_c,Client,User -0.008620943985647595 t_center_dist_c,Instructor -0.17606350643859145 t_dist_r,Course,CourseSession -0.008620943985647595 t_center_dist_r,User -0.008620943985647595 t_center_dist_c,Therapist -0.007121649379448013 t_dist_c,Instructor,Location -0.14655604775600914 t_dist_c,Therapist,CourseSession -0.008620943985647595 t_center_dist_r,CourseSession <= -2.5491448190601003
 newcut_72: -0.5463903637740334 t_zV_r,rt,Client,CourseSession -0.5463903637740334 t_zV_c,rt,Client,CourseSession -0.5463903637740334 t_zV_r,lb,Client,CourseSession -0.5463903637740334 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Course +0.7844437941013883 t_zV_c,lb,Course,Clinic +0.023705004389815622 t_x_c,Location -0.023705004389815622 t_x_c,Clinic -0.04741000877963124 t_x_c,CourseSession +0.001755926251097451 t_x_c,Course +0.045654082528533785 t_x_c,CertificateTemplate -0.023705004389815622 t_dist_c,Clinic,Location -0.11870061457418785 t_dist_r,CertificateTemplate,CourseSession -0.02027444476961597 t_dist_c,Course,CourseSession -0.023705004389815622 t_dist_c,Location,CourseSession -0.023705004389815622 t_dist_c,Client,Location -0.1458361785842031 t_dist_c,CertificateTemplate,CourseSession -0.08466072996362722 t_dist_c,Client,CourseSession -0.08466072996362722 t_dist_r,Client,CourseSession <= -1.4243321209080642
 newcut_73: -0.6052185575809308 t_zV_r,rt,Client,CertificateTemplate -0.6052185575809308 t_zV_r,lb,Client,CertificateTemplate -1.0 t_zV_c,rt,Client,User +0.6052185575809308 t_zV_r,lb,Client,User -1.0 t_zV_c,lb,Client,User +0.6052185575809308 t_zV_c,lb,Client,Clinic +0.6052185575809308 t_zV_c,lb,CertificateTemplate,User -0.6183064073645067 t_zV_r,rt,CertificateTemplate,CourseSession -0.6183064073645067 t_zV_c,rt,CertificateTemplate,CourseSession -0.6183064073645067 t_zV_r,lb,CertificateTemplate,CourseSession -0.6183064073645067 t_zV_c,lb,CertificateTemplate,CourseSession +0.015143571297933311 t_x_c,Location -0.030006706090349333 t_x_c,Clinic -0.01486313479241603 t_x_c,CourseSession +0.029726269584832054 t_x_c,CertificateTemplate -0.10032615984880816 t_x_r,User +0.10032615984880816 t_x_r,Client -0.02972626958483205 t_dist_c,User,Therapist -0.030006706090349333 t_dist_c,Clinic,Location -0.07728830092056334 t_dist_r,CertificateTemplate,CourseSession -0.30097847954642454 t_dist_r,Client,User -0.02972626958483205 t_dist_c,Clinic,Therapist -0.014863134792416022 t_dist_c,Location,CourseSession -0.03028714259586662 t_dist_c,Client,Location -0.07728830092056334 t_dist_c,CertificateTemplate,CourseSession -0.1395842223983418 t_dist_c,Client,CertificateTemplate -0.30097847954642454 t_dist_c,Client,User <= -3.004518075961715
 newcut_74: -0.11251904168397704 t_zV_r,rt,Client,CourseSession -0.11251904168397704 t_zV_c,rt,Client,CourseSession -0.11251904168397704 t_zV_r,lb,Client,CourseSession -0.11251904168397704 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,User +0.7760274815738044 t_zV_c,lb,User,Location -0.0096747141824001 t_x_c,Location -0.009674714182400101 t_x_c,Clinic -0.025519703334410424 t_x_c,CourseSession +0.0008309098462816799 t_x_c,User +0.04403822185292894 t_x_c,CertificateTemplate -0.019349428364800202 t_dist_c,User,Therapist -0.009674714182400101 t_dist_c,Clinic,Location -0.11449937681761527 t_dist_r,CertificateTemplate,CourseSession -0.019349428364800202 t_dist_c,Clinic,Therapist -0.02201911092646447 t_dist_c,Location,CourseSession -0.003500592407945952 t_dist_c,Client,Location -0.1400190801520257 t_dist_c,CertificateTemplate,CourseSession -0.012502115742664115 t_dist_c,Client,CourseSession -0.012502115742664115 t_dist_r,Client,CourseSession <= -0.34874517225992024
 newcut_75: -0.6618262806236077 t_zV_c,lb,Location,Instructor -0.6618262806236077 t_zV_r,lb,Location,Instructor -0.6618262806236077 t_zV_c,rt,Location,Instructor -0.6618262806236077 t_zV_r,rt,Location,Instructor +1.0 t_zV_r,rt,Course,CourseSession -0.7661215399300029 t_zV_r,rt,CourseSession,Location -0.7661215399300029 t_zV_c,rt,CourseSession,Location -0.7661215399300029 t_zV_r,lb,CourseSession,Location -0.7661215399300029 t_zV_c,lb,CourseSession,Location +1.4731403118040074 t_zV_r,lb,CourseSession,Instructor +0.017416481069042304 t_x_c,Location -0.03483296213808461 t_x_c,Clinic +0.06966592427616922 t_x_c,User +0.06966592427616922 t_x_c,CertificateTemplate -0.05224944320712692 t_x_c,Client -0.20899777282850768 t_x_r,Instructor +0.20899777282850765 t_x_r,Location +0.20899777282850768 t_x_r,CourseSession -0.13933184855233843 t_x_r,User -0.034832962138084615 t_x_r,Course -0.06966592427616922 t_x_r,Client -0.06966592427616922 t_x_c,Instructor -0.06966592427616922 t_center_dist_r,Clinic -0.06966592427616922 t_dist_r,Role,User -0.06966592427616922 t_dist_r,Clinic,Location -0.06966592427616922 t_dist_c,User,Therapist -0.06966592427616922 t_center_dist_c,Location -0.10449888641425382 t_dist_c,Clinic,Location -0.24880687241489002 t_dist_r,Location,CourseSession -0.06966592427616922 t_center_dist_r,Location -0.13933184855233843 t_dist_r,Client,User -0.09083890126206379 t_dist_c,Course,CourseSession -0.06966592427616922 t_center_dist_c,Clinic -0.06966592427616922 t_dist_c,Clinic,Therapist -0.24880687241489002 t_dist_c,Location,CourseSession -0.400579064587973 t_dist_r,Client,Instructor -0.06966592427616922 t_center_dist_c,Role -0.06966592427616922 t_center_dist_r,Instructor -0.06966592427616922 t_center_dist_c,CertificateTemplate -0.06966592427616922 t_center_dist_c,Course -0.06966592427616922 t_dist_c,Role,User -0.06966592427616922 t_center_dist_c,CourseSession -0.06966592427616922 t_center_dist_r,Therapist -0.26124721603563456 t_dist_c,Client,Location -0.20899777282850765 t_dist_r,Client,Location -0.13933184855233843 t_dist_r,Instructor,Location -0.10449888641425384 t_center_dist_r,Course -0.06966592427616922 t_center_dist_c,User -0.034832962138084615 t_dist_c,CertificateTemplate,CourseSession -0.10449888641425384 t_dist_c,CourseSession,Instructor -0.03483296213808461 t_dist_c,Client,CertificateTemplate -0.06966592427616922 t_center_dist_r,Client -0.06966592427616922 t_center_dist_r,CertificateTemplate -0.06966592427616922 t_center_dist_r,Role -0.06966592427616922 t_center_dist_c,Client -0.47024498886414223 t_dist_c,Client,Instructor -0.20899777282850765 t_dist_c,Client,User -0.06966592427616922 t_center_dist_c,Instructor -0.09083890126206379 t_dist_r,Course,CourseSession -0.06966592427616922 t_center_dist_r,User -0.06966592427616922 t_center_dist_c,Therapist -0.1620489977728284 t_dist_c,Instructor,Location -0.06966592427616924 t_dist_r,CourseSession,Instructor -0.06966592427616922 t_center_dist_r,CourseSession <= -12.159793191218572
 newcut_76: +1.0 t_zV_c,lb,Course,User -0.8135593220338979 t_zV_r,rt,Course,CourseSession -0.8135593220338979 t_zV_c,rt,Course,CourseSession -0.8135593220338979 t_zV_r,lb,Course,CourseSession -0.8135593220338979 t_zV_c,lb,Course,CourseSession -1.115254237288135 t_zV_c,rt,Role,User +1.0 t_zV_r,lb,Role,User -1.115254237288135 t_zV_c,lb,Role,User -0.020338983050847453 t_x_c,CourseSession -0.040677966101694885 t_x_c,User +0.020338983050847432 t_x_c,Role +0.040677966101694905 t_x_c,Course -0.04406779661016948 t_x_r,User +0.04406779661016948 t_x_r,Role -0.3694915254237287 t_dist_r,Role,User -0.1084745762711864 t_dist_c,Course,CourseSession -0.020338983050847453 t_center_dist_c,Role -0.3694915254237287 t_dist_c,Role,User -0.020338983050847453 t_center_dist_c,CourseSession -0.1084745762711864 t_dist_r,Course,CourseSession <= -2.6593220338983037
Bounds
 0 <= t_zV_r,lb,Therapist,Instructor <= 1
 0 <= t_zV_c,rt,Therapist,Instructor <= 1
 0 <= t_zV_r,rt,Therapist,Instructor <= 1
 0 <= t_zV_c,lb,Location,Instructor <= 1
 0 <= t_zV_r,lb,Location,Instructor <= 1
 0 <= t_zV_c,rt,Location,Instructor <= 1
 0 <= t_zV_r,rt,Location,Instructor <= 1
 0 <= t_zV_c,lb,Location,Therapist <= 1
 0 <= t_zV_r,lb,Location,Therapist <= 1
 0 <= t_zV_c,rt,Location,Therapist <= 1
 0 <= t_zV_r,rt,Location,Therapist <= 1
 0 <= t_zV_c,lb,Clinic,Instructor <= 1
 0 <= t_zV_r,lb,Clinic,Instructor <= 1
 0 <= t_zV_c,rt,Clinic,Instructor <= 1
 0 <= t_zV_r,rt,Clinic,Instructor <= 1
 0 <= t_zV_c,lb,Clinic,Therapist <= 1
 0 <= t_zV_r,lb,Clinic,Therapist <= 1
 0 <= t_zV_c,rt,Clinic,Therapist <= 1
 0 <= t_zV_r,rt,Clinic,Therapist <= 1
 0 <= t_zV_c,lb,Clinic,Location <= 1
 0 <= t_zV_r,lb,Clinic,Location <= 1
 0 <= t_zV_r,rt,Client,CertificateTemplate <= 1
 0 <= t_zV_c,rt,Client,CertificateTemplate <= 1
 0 <= t_zV_r,lb,Client,CertificateTemplate <= 1
 0 <= t_zV_c,lb,Client,CertificateTemplate <= 1
 0 <= t_zV_r,rt,Client,Course <= 1
 0 <= t_zV_c,rt,Client,Course <= 1
 0 <= t_zV_r,lb,Client,Course <= 1
 0 <= t_zV_c,lb,Client,Course <= 1
 0 <= t_zV_r,rt,Client,Role <= 1
 0 <= t_zV_c,rt,Client,Role <= 1
 0 <= t_zV_r,lb,Client,Role <= 1
 0 <= t_zV_c,lb,Client,Role <= 1
 0 <= t_zV_r,rt,Client,User <= 1
 0 <= t_zV_c,rt,Client,User <= 1
 0 <= t_zV_r,lb,Client,User <= 1
 0 <= t_zV_c,lb,Client,User <= 1
 0 <= t_zV_r,rt,Client,CourseSession <= 1
 0 <= t_zV_c,rt,Client,CourseSession <= 1
 0 <= t_zV_r,lb,Client,CourseSession <= 1
 0 <= t_zV_c,lb,Client,CourseSession <= 1
 0 <= t_zV_r,rt,Client,Clinic <= 1
 0 <= t_zV_c,rt,Client,Clinic <= 1
 0 <= t_zV_r,lb,Client,Clinic <= 1
 0 <= t_zV_c,lb,Client,Clinic <= 1
 0 <= t_zV_r,rt,Client,Location <= 1
 0 <= t_zV_c,rt,Client,Location <= 1
 0 <= t_zV_r,lb,Client,Location <= 1
 0 <= t_zV_c,lb,Client,Location <= 1
 0 <= t_zV_r,rt,Client,Therapist <= 1
 0 <= t_zV_c,rt,Client,Therapist <= 1
 0 <= t_zV_r,lb,Client,Therapist <= 1
 0 <= t_zV_c,lb,Client,Therapist <= 1
 0 <= t_zV_r,rt,Client,Instructor <= 1
 0 <= t_zV_c,rt,Client,Instructor <= 1
 0 <= t_zV_r,lb,Client,Instructor <= 1
 0 <= t_zV_c,lb,Client,Instructor <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Course <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Course <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Course <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Course <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Role <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Role <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Role <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Role <= 1
 0 <= t_zV_r,rt,CertificateTemplate,User <= 1
 0 <= t_zV_c,rt,CertificateTemplate,User <= 1
 0 <= t_zV_r,lb,CertificateTemplate,User <= 1
 0 <= t_zV_c,lb,CertificateTemplate,User <= 1
 0 <= t_zV_r,rt,CertificateTemplate,CourseSession <= 1
 0 <= t_zV_c,rt,CertificateTemplate,CourseSession <= 1
 0 <= t_zV_r,lb,CertificateTemplate,CourseSession <= 1
 0 <= t_zV_c,lb,CertificateTemplate,CourseSession <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Clinic <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Clinic <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Clinic <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Clinic <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Location <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Location <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Location <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Location <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Therapist <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Therapist <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Therapist <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Therapist <= 1
 0 <= t_zV_r,rt,CertificateTemplate,Instructor <= 1
 0 <= t_zV_c,rt,CertificateTemplate,Instructor <= 1
 0 <= t_zV_r,lb,CertificateTemplate,Instructor <= 1
 0 <= t_zV_c,lb,CertificateTemplate,Instructor <= 1
 0 <= t_zV_r,rt,Course,Role <= 1
 0 <= t_zV_c,rt,Course,Role <= 1
 0 <= t_zV_r,lb,Course,Role <= 1
 0 <= t_zV_c,lb,Course,Role <= 1
 0 <= t_zV_r,rt,Course,User <= 1
 0 <= t_zV_c,rt,Course,User <= 1
 0 <= t_zV_r,lb,Course,User <= 1
 0 <= t_zV_c,lb,Course,User <= 1
 0 <= t_zV_r,rt,Course,CourseSession <= 1
 0 <= t_zV_c,rt,Course,CourseSession <= 1
 0 <= t_zV_r,lb,Course,CourseSession <= 1
 0 <= t_zV_c,lb,Course,CourseSession <= 1
 0 <= t_zV_r,rt,Course,Clinic <= 1
 0 <= t_zV_c,rt,Course,Clinic <= 1
 0 <= t_zV_r,lb,Course,Clinic <= 1
 0 <= t_zV_c,lb,Course,Clinic <= 1
 0 <= t_zV_r,rt,Course,Location <= 1
 0 <= t_zV_c,rt,Course,Location <= 1
 0 <= t_zV_r,lb,Course,Location <= 1
 0 <= t_zV_c,lb,Course,Location <= 1
 0 <= t_zV_r,rt,Course,Therapist <= 1
 0 <= t_zV_c,rt,Course,Therapist <= 1
 0 <= t_zV_r,lb,Course,Therapist <= 1
 0 <= t_zV_c,lb,Course,Therapist <= 1
 0 <= t_zV_r,rt,Course,Instructor <= 1
 0 <= t_zV_c,rt,Course,Instructor <= 1
 0 <= t_zV_r,lb,Course,Instructor <= 1
 0 <= t_zV_c,lb,Course,Instructor <= 1
 0 <= t_zV_r,rt,Role,User <= 1
 0 <= t_zV_c,rt,Role,User <= 1
 0 <= t_zV_r,lb,Role,User <= 1
 0 <= t_zV_c,lb,Role,User <= 1
 0 <= t_zV_r,rt,Role,CourseSession <= 1
 0 <= t_zV_c,rt,Role,CourseSession <= 1
 0 <= t_zV_r,lb,Role,CourseSession <= 1
 0 <= t_zV_c,lb,Role,CourseSession <= 1
 0 <= t_zV_r,rt,Role,Clinic <= 1
 0 <= t_zV_c,rt,Role,Clinic <= 1
 0 <= t_zV_r,lb,Role,Clinic <= 1
 0 <= t_zV_c,lb,Role,Clinic <= 1
 0 <= t_zV_r,rt,Role,Location <= 1
 0 <= t_zV_c,rt,Role,Location <= 1
 0 <= t_zV_r,lb,Role,Location <= 1
 0 <= t_zV_c,lb,Role,Location <= 1
 0 <= t_zV_r,rt,Role,Therapist <= 1
 0 <= t_zV_c,rt,Role,Therapist <= 1
 0 <= t_zV_r,lb,Role,Therapist <= 1
 0 <= t_zV_c,lb,Role,Therapist <= 1
 0 <= t_zV_r,rt,Role,Instructor <= 1
 0 <= t_zV_c,rt,Role,Instructor <= 1
 0 <= t_zV_r,lb,Role,Instructor <= 1
 0 <= t_zV_c,lb,Role,Instructor <= 1
 0 <= t_zV_r,rt,User,CourseSession <= 1
 0 <= t_zV_c,rt,User,CourseSession <= 1
 0 <= t_zV_r,lb,User,CourseSession <= 1
 0 <= t_zV_c,lb,User,CourseSession <= 1
 0 <= t_zV_r,rt,User,Clinic <= 1
 0 <= t_zV_c,rt,User,Clinic <= 1
 0 <= t_zV_r,lb,User,Clinic <= 1
 0 <= t_zV_c,lb,User,Clinic <= 1
 0 <= t_zV_r,rt,User,Location <= 1
 0 <= t_zV_c,rt,User,Location <= 1
 0 <= t_zV_r,lb,User,Location <= 1
 0 <= t_zV_c,lb,User,Location <= 1
 0 <= t_zV_r,rt,User,Therapist <= 1
 0 <= t_zV_c,rt,User,Therapist <= 1
 0 <= t_zV_r,lb,User,Therapist <= 1
 0 <= t_zV_c,lb,User,Therapist <= 1
 0 <= t_zV_r,rt,User,Instructor <= 1
 0 <= t_zV_c,rt,User,Instructor <= 1
 0 <= t_zV_r,lb,User,Instructor <= 1
 0 <= t_zV_c,lb,User,Instructor <= 1
 0 <= t_zV_r,rt,CourseSession,Clinic <= 1
 0 <= t_zV_c,rt,CourseSession,Clinic <= 1
 0 <= t_zV_r,lb,CourseSession,Clinic <= 1
 0 <= t_zV_c,lb,CourseSession,Clinic <= 1
 0 <= t_zV_r,rt,CourseSession,Location <= 1
 0 <= t_zV_c,rt,CourseSession,Location <= 1
 0 <= t_zV_r,lb,CourseSession,Location <= 1
 0 <= t_zV_c,lb,CourseSession,Location <= 1
 0 <= t_zV_r,rt,CourseSession,Therapist <= 1
 0 <= t_zV_c,rt,CourseSession,Therapist <= 1
 0 <= t_zV_r,lb,CourseSession,Therapist <= 1
 0 <= t_zV_c,lb,CourseSession,Therapist <= 1
 0 <= t_zV_r,rt,CourseSession,Instructor <= 1
 0 <= t_zV_c,rt,CourseSession,Instructor <= 1
 0 <= t_zV_r,lb,CourseSession,Instructor <= 1
 0 <= t_zV_c,lb,CourseSession,Instructor <= 1
 0 <= t_zV_r,rt,Clinic,Location <= 1
 0 <= t_zV_c,rt,Clinic,Location <= 1
 0 <= t_zV_c,lb,Therapist,Instructor <= 1
 1 <= t_x_c,Therapist <= 32
 1 <= t_x_c,Location <= 32
 1 <= t_x_c,Clinic <= 32
 1 <= t_x_c,CourseSession <= 29
 1 <= t_x_c,User <= 32
 1 <= t_x_c,Role <= 33
 1 <= t_x_c,Course <= 33
 1 <= t_x_c,CertificateTemplate <= 31
 1 <= t_x_c,Client <= 32
 1 <= t_x_r,Instructor <= 33
 1 <= t_x_r,Therapist <= 33
 1 <= t_x_r,Location <= 34
 1 <= t_x_r,Clinic <= 35
 1 <= t_x_r,CourseSession <= 34
 1 <= t_x_r,User <= 36
 1 <= t_x_r,Role <= 37
 1 <= t_x_r,Course <= 37
 1 <= t_x_r,CertificateTemplate <= 36
 1 <= t_x_r,Client <= 34
 1 <= t_x_c,Instructor <= 31
 0.5 <= t_center_dist_c,Therapist
 t_center_dist_c,Instructor free
 0 <= t_dist_r,Client,CertificateTemplate <= 43
 0 <= t_dist_r,Client,CourseSession <= 42
 0 <= t_dist_r,Client,Instructor <= 42.5
 0 <= t_dist_r,Client,Location <= 42
 0 <= t_dist_r,Client,User <= 43
 0 <= t_dist_r,CertificateTemplate,CourseSession <= 43
 0 <= t_dist_r,Course,CourseSession <= 43.5
 0 <= t_dist_r,CourseSession,Instructor <= 42.5
 0 <= t_dist_r,Instructor,Location <= 42.5
 0 <= t_dist_r,Clinic,Location <= 42.5
 0 <= t_dist_r,Clinic,Therapist <= 43
 0 <= t_dist_r,Location,Therapist <= 42.5
 0 <= t_dist_r,Location,CourseSession <= 42
 0 <= t_dist_r,Therapist,CourseSession <= 42.5
 0 <= t_dist_r,Role,User <= 42.5
 0 <= t_dist_r,User,Instructor <= 43.5
 0 <= t_dist_r,User,Therapist <= 43.5
 0 <= t_dist_c,Client,CertificateTemplate <= 42.5
 0 <= t_dist_c,Client,CourseSession <= 43.5
 0 <= t_dist_c,Client,Instructor <= 42.5
 0 <= t_dist_c,Client,Location <= 42
 0 <= t_dist_c,Client,User <= 42
 0 <= t_dist_c,CertificateTemplate,CourseSession <= 43
 0 <= t_dist_c,Course,CourseSession <= 44
 0 <= t_dist_c,CourseSession,Instructor <= 43
 0 <= t_dist_c,Instructor,Location <= 42.5
 0 <= t_dist_c,Clinic,Location <= 42
 0 <= t_dist_c,Clinic,Therapist <= 42
 0 <= t_dist_c,Location,Therapist <= 42
 0 <= t_dist_c,Location,CourseSession <= 43.5
 0 <= t_dist_c,Therapist,CourseSession <= 43.5
 0 <= t_dist_c,Role,User <= 42.5
 0 <= t_dist_c,User,Instructor <= 42.5
 0 <= t_dist_c,User,Therapist <= 42
 t_center_dist_r,Client free
 t_center_dist_r,CertificateTemplate free
 t_center_dist_r,Course free
 t_center_dist_r,Role free
 t_center_dist_r,User free
 0.5 <= t_center_dist_r,CourseSession
 t_center_dist_r,Clinic free
 0.5 <= t_center_dist_r,Location
 t_center_dist_r,Therapist free
 t_center_dist_r,Instructor free
 0.5 <= t_center_dist_c,Client
 t_center_dist_c,CertificateTemplate free
 t_center_dist_c,Course free
 t_center_dist_c,Role free
 0.5 <= t_center_dist_c,User
 t_center_dist_c,CourseSession free
 0.5 <= t_center_dist_c,Clinic
 0.5 <= t_center_dist_c,Location
Binaries
 t_zV_r,lb,Therapist,Instructor t_zV_c,rt,Therapist,Instructor t_zV_r,rt,Therapist,Instructor t_zV_c,lb,Location,Instructor
 t_zV_r,lb,Location,Instructor t_zV_c,rt,Location,Instructor t_zV_r,rt,Location,Instructor t_zV_c,lb,Location,Therapist
 t_zV_r,lb,Location,Therapist t_zV_c,rt,Location,Therapist t_zV_r,rt,Location,Therapist t_zV_c,lb,Clinic,Instructor
 t_zV_r,lb,Clinic,Instructor t_zV_c,rt,Clinic,Instructor t_zV_r,rt,Clinic,Instructor t_zV_c,lb,Clinic,Therapist
 t_zV_r,lb,Clinic,Therapist t_zV_c,rt,Clinic,Therapist t_zV_r,rt,Clinic,Therapist t_zV_c,lb,Clinic,Location
 t_zV_r,lb,Clinic,Location t_zV_r,rt,Client,CertificateTemplate t_zV_c,rt,Client,CertificateTemplate
 t_zV_r,lb,Client,CertificateTemplate t_zV_c,lb,Client,CertificateTemplate t_zV_r,rt,Client,Course
 t_zV_c,rt,Client,Course t_zV_r,lb,Client,Course t_zV_c,lb,Client,Course t_zV_r,rt,Client,Role t_zV_c,rt,Client,Role
 t_zV_r,lb,Client,Role t_zV_c,lb,Client,Role t_zV_r,rt,Client,User t_zV_c,rt,Client,User t_zV_r,lb,Client,User
 t_zV_c,lb,Client,User t_zV_r,rt,Client,CourseSession t_zV_c,rt,Client,CourseSession t_zV_r,lb,Client,CourseSession
 t_zV_c,lb,Client,CourseSession t_zV_r,rt,Client,Clinic t_zV_c,rt,Client,Clinic t_zV_r,lb,Client,Clinic
 t_zV_c,lb,Client,Clinic t_zV_r,rt,Client,Location t_zV_c,rt,Client,Location t_zV_r,lb,Client,Location
 t_zV_c,lb,Client,Location t_zV_r,rt,Client,Therapist t_zV_c,rt,Client,Therapist t_zV_r,lb,Client,Therapist
 t_zV_c,lb,Client,Therapist t_zV_r,rt,Client,Instructor t_zV_c,rt,Client,Instructor t_zV_r,lb,Client,Instructor
 t_zV_c,lb,Client,Instructor t_zV_r,rt,CertificateTemplate,Course t_zV_c,rt,CertificateTemplate,Course
 t_zV_r,lb,CertificateTemplate,Course t_zV_c,lb,CertificateTemplate,Course t_zV_r,rt,CertificateTemplate,Role
 t_zV_c,rt,CertificateTemplate,Role t_zV_r,lb,CertificateTemplate,Role t_zV_c,lb,CertificateTemplate,Role
 t_zV_r,rt,CertificateTemplate,User t_zV_c,rt,CertificateTemplate,User t_zV_r,lb,CertificateTemplate,User
 t_zV_c,lb,CertificateTemplate,User t_zV_r,rt,CertificateTemplate,CourseSession t_zV_c,rt,CertificateTemplate,CourseSession
 t_zV_r,lb,CertificateTemplate,CourseSession t_zV_c,lb,CertificateTemplate,CourseSession t_zV_r,rt,CertificateTemplate,Clinic
 t_zV_c,rt,CertificateTemplate,Clinic t_zV_r,lb,CertificateTemplate,Clinic t_zV_c,lb,CertificateTemplate,Clinic
 t_zV_r,rt,CertificateTemplate,Location t_zV_c,rt,CertificateTemplate,Location t_zV_r,lb,CertificateTemplate,Location
 t_zV_c,lb,CertificateTemplate,Location t_zV_r,rt,CertificateTemplate,Therapist t_zV_c,rt,CertificateTemplate,Therapist
 t_zV_r,lb,CertificateTemplate,Therapist t_zV_c,lb,CertificateTemplate,Therapist t_zV_r,rt,CertificateTemplate,Instructor
 t_zV_c,rt,CertificateTemplate,Instructor t_zV_r,lb,CertificateTemplate,Instructor t_zV_c,lb,CertificateTemplate,Instructor
 t_zV_r,rt,Course,Role t_zV_c,rt,Course,Role t_zV_r,lb,Course,Role t_zV_c,lb,Course,Role t_zV_r,rt,Course,User
 t_zV_c,rt,Course,User t_zV_r,lb,Course,User t_zV_c,lb,Course,User t_zV_r,rt,Course,CourseSession
 t_zV_c,rt,Course,CourseSession t_zV_r,lb,Course,CourseSession t_zV_c,lb,Course,CourseSession t_zV_r,rt,Course,Clinic
 t_zV_c,rt,Course,Clinic t_zV_r,lb,Course,Clinic t_zV_c,lb,Course,Clinic t_zV_r,rt,Course,Location
 t_zV_c,rt,Course,Location t_zV_r,lb,Course,Location t_zV_c,lb,Course,Location t_zV_r,rt,Course,Therapist
 t_zV_c,rt,Course,Therapist t_zV_r,lb,Course,Therapist t_zV_c,lb,Course,Therapist t_zV_r,rt,Course,Instructor
 t_zV_c,rt,Course,Instructor t_zV_r,lb,Course,Instructor t_zV_c,lb,Course,Instructor t_zV_r,rt,Role,User
 t_zV_c,rt,Role,User t_zV_r,lb,Role,User t_zV_c,lb,Role,User t_zV_r,rt,Role,CourseSession t_zV_c,rt,Role,CourseSession
 t_zV_r,lb,Role,CourseSession t_zV_c,lb,Role,CourseSession t_zV_r,rt,Role,Clinic t_zV_c,rt,Role,Clinic
 t_zV_r,lb,Role,Clinic t_zV_c,lb,Role,Clinic t_zV_r,rt,Role,Location t_zV_c,rt,Role,Location t_zV_r,lb,Role,Location
 t_zV_c,lb,Role,Location t_zV_r,rt,Role,Therapist t_zV_c,rt,Role,Therapist t_zV_r,lb,Role,Therapist
 t_zV_c,lb,Role,Therapist t_zV_r,rt,Role,Instructor t_zV_c,rt,Role,Instructor t_zV_r,lb,Role,Instructor
 t_zV_c,lb,Role,Instructor t_zV_r,rt,User,CourseSession t_zV_c,rt,User,CourseSession t_zV_r,lb,User,CourseSession
 t_zV_c,lb,User,CourseSession t_zV_r,rt,User,Clinic t_zV_c,rt,User,Clinic t_zV_r,lb,User,Clinic t_zV_c,lb,User,Clinic
 t_zV_r,rt,User,Location t_zV_c,rt,User,Location t_zV_r,lb,User,Location t_zV_c,lb,User,Location t_zV_r,rt,User,Therapist
 t_zV_c,rt,User,Therapist t_zV_r,lb,User,Therapist t_zV_c,lb,User,Therapist t_zV_r,rt,User,Instructor
 t_zV_c,rt,User,Instructor t_zV_r,lb,User,Instructor t_zV_c,lb,User,Instructor t_zV_r,rt,CourseSession,Clinic
 t_zV_c,rt,CourseSession,Clinic t_zV_r,lb,CourseSession,Clinic t_zV_c,lb,CourseSession,Clinic t_zV_r,rt,CourseSession,Location
 t_zV_c,rt,CourseSession,Location t_zV_r,lb,CourseSession,Location t_zV_c,lb,CourseSession,Location
 t_zV_r,rt,CourseSession,Therapist t_zV_c,rt,CourseSession,Therapist t_zV_r,lb,CourseSession,Therapist
 t_zV_c,lb,CourseSession,Therapist t_zV_r,rt,CourseSession,Instructor t_zV_c,rt,CourseSession,Instructor
 t_zV_r,lb,CourseSession,Instructor t_zV_c,lb,CourseSession,Instructor t_zV_r,rt,Clinic,Location t_zV_c,rt,Clinic,Location
 t_zV_c,lb,Therapist,Instructor
Generals
 t_x_c,Therapist t_x_c,Location t_x_c,Clinic t_x_c,CourseSession t_x_c,User t_x_c,Role t_x_c,Course
 t_x_c,CertificateTemplate t_x_c,Client t_x_r,Instructor t_x_r,Therapist t_x_r,Location t_x_r,Clinic
 t_x_r,CourseSession t_x_r,User t_x_r,Role t_x_r,Course t_x_r,CertificateTemplate t_x_r,Client t_x_c,Instructor
End
