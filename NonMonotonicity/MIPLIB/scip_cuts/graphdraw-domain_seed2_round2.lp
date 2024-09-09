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
 newcut_1: +1.0 t_zV_r,lb,Role,User -0.07692307692307693 t_x_r,User +0.07692307692307693 t_x_r,Role -0.07692307692307693 t_dist_r,Role,User -0.07692307692307693 t_dist_c,Role,User <= 0.038461538461538464
 newcut_2: -0.09373433583959873 t_zV_c,lb,Clinic,Location -0.09373433583959873 t_zV_r,lb,Clinic,Location +1.0 t_zV_r,lb,CertificateTemplate,CourseSession +1.0 t_zV_r,lb,CourseSession,Location -0.09373433583959873 t_zV_r,rt,Clinic,Location -0.09373433583959873 t_zV_c,rt,Clinic,Location +0.0025062656641603944 t_x_c,Therapist +0.0025062656641603944 t_x_c,Location -0.0025062656641603944 t_x_c,Clinic -0.0025062656641603944 t_x_c,User +0.0025062656641603957 t_x_c,Client +0.06390977443609018 t_x_r,CertificateTemplate -0.06390977443609018 t_x_r,Client -0.0025062656641603957 t_x_c,Instructor -0.011027568922305733 t_dist_r,Clinic,Location -0.0025062656641603944 t_dist_c,User,Therapist -0.011027568922305733 t_dist_c,Clinic,Location -0.009189640768588116 t_dist_r,Client,User -0.002506265664160395 t_dist_c,Clinic,Therapist -0.061403508771929786 t_dist_c,Client,Location -0.061403508771929786 t_dist_r,Client,Location -0.01441102756892227 t_dist_r,Instructor,Location -0.0025062656641603957 t_dist_c,CourseSession,Instructor -0.06390977443609018 t_dist_c,Client,CertificateTemplate -0.009189640768588116 t_dist_c,Client,User -0.06390977443609018 t_dist_r,Client,CertificateTemplate -0.0025062656641603957 t_dist_c,Client,CourseSession -0.01441102756892227 t_dist_c,Instructor,Location -0.0025062656641603957 t_dist_r,Client,CourseSession <= 0.09208437761069455
 newcut_3: +0.5195530726256985 t_zV_r,lb,Course,CourseSession +0.5195530726256985 t_zV_r,lb,CourseSession,Instructor -0.014897579143389201 t_x_r,Instructor +0.015828677839851025 t_x_r,Course -0.014897579143389201 t_center_dist_r,Instructor -0.015828677839851025 t_center_dist_r,Course -0.0018621973929236488 t_center_dist_r,CourseSession <= 0.5661080074487896
 newcut_4: +0.5373913043478262 t_zV_r,rt,Location,Instructor -0.021681159420289777 t_zV_c,lb,Clinic,Location -0.021681159420289774 t_zV_r,lb,Clinic,Location +0.5157101449275362 t_zV_r,rt,Clinic,Location -0.021681159420289777 t_zV_c,rt,Clinic,Location +0.0005797101449275342 t_x_c,Therapist +0.0005797101449275342 t_x_c,Location -0.0005797101449275342 t_x_c,Clinic -0.0005797101449275342 t_x_c,User +0.0005797101449275342 t_x_c,Client +0.014782608695652176 t_x_r,Instructor +0.0005797101449275324 t_x_r,Location -0.015072463768115942 t_x_r,Clinic +0.0005797101449275342 t_x_r,CourseSession -0.0005797101449275342 t_x_r,Client -0.0005797101449275342 t_x_c,Instructor -0.015072463768115942 t_center_dist_r,Clinic -0.0025507246376811504 t_dist_r,Clinic,Location -0.0005797101449275342 t_dist_c,User,Therapist -0.00255072463768115 t_dist_c,Clinic,Location -0.002125603864734292 t_dist_r,Client,User -0.0005797101449275342 t_dist_c,Clinic,Therapist -0.014782608695652176 t_center_dist_r,Instructor -0.0005797101449275342 t_dist_c,Client,Location -0.0005797101449275342 t_dist_r,Client,Location -0.0033333333333333223 t_dist_r,Instructor,Location -0.0005797101449275342 t_dist_c,CourseSession,Instructor -0.002125603864734292 t_dist_c,Client,User -0.0005797101449275342 t_dist_c,Client,CourseSession -0.0033333333333333223 t_dist_c,Instructor,Location -0.0005797101449275342 t_dist_r,Client,CourseSession -0.0005797101449275342 t_center_dist_r,CourseSession <= 0.4305748792270537
 newcut_5: -0.03282752403846143 t_zV_c,lb,Location,Instructor -0.032827524038461425 t_zV_r,lb,Location,Instructor -0.03282752403846143 t_zV_c,rt,Location,Instructor -0.032827524038461425 t_zV_r,rt,Location,Instructor +0.59375 t_zV_r,rt,Clinic,Therapist -0.01762820512820507 t_zV_r,rt,Client,User -0.017628205128205066 t_zV_c,rt,Client,User -0.01762820512820507 t_zV_r,lb,Client,User -0.017628205128205066 t_zV_c,lb,Client,User +0.59375 t_zV_r,lb,CourseSession,Therapist +0.0006009615384615363 t_x_c,Location -0.0006009615384615363 t_x_c,User +0.0006009615384615363 t_x_c,Client -0.0006009615384615363 t_x_r,Therapist +0.0006009615384615363 t_x_r,Location -0.015324519230769232 t_x_r,Clinic +0.03125 t_x_r,CourseSession -0.0006009615384615363 t_x_r,Client -0.0006009615384615363 t_x_c,Instructor -0.015324519230769232 t_center_dist_r,Clinic -0.0006009615384615363 t_dist_c,User,Therapist -0.002203525641025633 t_dist_r,Client,User -0.0006009615384615363 t_dist_c,Location,Therapist -0.0006009615384615363 t_dist_c,Client,Location -0.0006009615384615363 t_dist_r,Client,Location -0.0034555288461538347 t_dist_r,Instructor,Location -0.0006009615384615363 t_dist_r,Location,Therapist -0.0006009615384615363 t_dist_c,CourseSession,Instructor -0.002203525641025633 t_dist_c,Client,User -0.0006009615384615363 t_dist_c,Client,CourseSession -0.0034555288461538343 t_dist_c,Instructor,Location -0.0006009615384615363 t_dist_r,Client,CourseSession -0.030649038461538464 t_center_dist_r,CourseSession <= 0.7485476762820519
 newcut_6: +1.0 t_zV_c,lb,Course,Role +1.0 t_zV_c,lb,Role,Instructor +0.023809523809523815 t_x_c,Course -0.023809523809523815 t_x_c,Instructor -0.023809523809523815 t_center_dist_c,Course -0.023809523809523815 t_center_dist_c,Instructor <= 1.0238095238095237
 newcut_7: -0.03410864814236671 t_zV_c,lb,Location,Instructor -0.0341086481423667 t_zV_r,lb,Location,Instructor -0.03410864814236671 t_zV_c,rt,Location,Instructor -0.0341086481423667 t_zV_r,rt,Location,Instructor +0.7208866687480491 t_zV_c,lb,Course,Therapist +0.7208866687480489 t_zV_c,lb,Therapist,Instructor +0.0006244146113019075 t_x_c,Location -0.0006244146113019075 t_x_c,Clinic +0.01654698719950047 t_x_c,Course -0.01654698719950047 t_x_c,Instructor -0.002747424289728393 t_dist_r,Clinic,Location -0.002747424289728393 t_dist_c,Clinic,Location -0.0006244146113019075 t_dist_c,Clinic,Therapist -0.01654698719950047 t_center_dist_c,Course -0.0035903840149859693 t_dist_r,Instructor,Location -0.01654698719950047 t_center_dist_c,Instructor -0.0035903840149859685 t_dist_c,Instructor,Location <= 0.6458632532001246
 newcut_8: -0.03525330751855455 t_zV_c,lb,Location,Instructor -0.03525330751855455 t_zV_r,lb,Location,Instructor -0.03525330751855455 t_zV_c,rt,Location,Instructor -0.03525330751855455 t_zV_r,rt,Location,Instructor +0.77863827040981 t_zV_c,lb,Course,User +0.7786382704098098 t_zV_c,lb,User,CourseSession +0.0006453694740238819 t_x_c,Therapist +0.0006453694740238819 t_x_c,Location -0.0006453694740238819 t_x_c,Clinic -0.0006453694740238825 t_x_c,User +0.017102291061632788 t_x_c,Course -0.017102291061632788 t_x_c,Instructor -0.0028396256857050797 t_dist_r,Clinic,Location -0.0006453694740238819 t_dist_c,User,Therapist -0.0028396256857050797 t_dist_c,Clinic,Location -0.0006453694740238819 t_dist_c,Clinic,Therapist -0.017102291061632788 t_center_dist_c,Course -0.0037108744756373215 t_dist_r,Instructor,Location -0.03355921264924169 t_dist_c,CourseSession,Instructor -0.017102291061632788 t_center_dist_c,Instructor -0.0037108744756373215 t_dist_c,Instructor,Location <= 0.7010971281058404
 newcut_9: -0.2917878410192491 t_zV_r,rt,Client,User -0.2917878410192491 t_zV_c,rt,Client,User -0.2917878410192491 t_zV_r,lb,Client,User -0.2917878410192491 t_zV_c,lb,Client,User +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_c,lb,User,Clinic +0.044869131699210626 t_x_c,Therapist -0.04486913169921062 t_x_c,Clinic -0.044038221852928945 t_x_c,User +0.02201911092646447 t_x_c,CertificateTemplate +0.02201911092646447 t_x_c,Client -0.044869131699210626 t_dist_c,User,Therapist -0.16147348012740614 t_dist_r,Client,User -0.044869131699210626 t_dist_c,Clinic,Therapist -0.02201911092646447 t_dist_c,Client,CertificateTemplate -0.16147348012740614 t_dist_c,Client,User <= -0.5945852375017303
 newcut_10: -0.11316187594553721 t_zV_c,lb,Clinic,Location -0.11316187594553721 t_zV_r,lb,Clinic,Location +1.0 t_zV_c,lb,Role,CourseSession +1.0 t_zV_c,lb,CourseSession,Clinic -0.11316187594553721 t_zV_r,rt,Clinic,Location -0.11316187594553721 t_zV_c,rt,Clinic,Location +0.04236006051437216 t_x_c,Therapist +0.0030257186081694446 t_x_c,Location -0.04236006051437217 t_x_c,Clinic -0.02118003025718608 t_x_c,User +0.02118003025718608 t_x_c,Role -0.0030257186081694446 t_x_c,Instructor -0.013313161875945554 t_dist_r,Clinic,Location -0.04236006051437216 t_dist_c,User,Therapist -0.013313161875945554 t_dist_c,Clinic,Location -0.04236006051437216 t_dist_c,Clinic,Therapist -0.02118003025718608 t_dist_c,Role,User -0.017397881996974306 t_dist_r,Instructor,Location -0.0030257186081694446 t_dist_c,CourseSession,Instructor -0.017397881996974306 t_dist_c,Instructor,Location <= 0.6204992435703474
 newcut_11: +1.0 t_zV_r,lb,Course,User -0.03657289970847689 t_x_c,CourseSession +0.03657289970847689 t_x_c,User +0.03657289970847689 t_x_c,CertificateTemplate -0.03657289970847689 t_x_c,Client -0.03657289970847689 t_x_r,Therapist +0.03657289970847689 t_x_r,Location +0.07314579941695377 t_x_r,CourseSession -0.07314579941695377 t_x_r,User +0.05485934956271533 t_x_r,Course -0.07314579941695377 t_x_r,Client -0.03657289970847689 t_center_dist_r,Clinic -0.03657289970847689 t_dist_r,Role,User -0.03657289970847689 t_dist_c,User,Therapist -0.03657289970847689 t_center_dist_c,Location -0.03657289970847689 t_dist_r,Therapist,CourseSession -0.0950895392420399 t_dist_r,CertificateTemplate,CourseSession -0.03657289970847689 t_center_dist_r,Location -0.03657289970847689 t_center_dist_c,Clinic -0.03657289970847689 t_center_dist_c,Role -0.03657289970847689 t_center_dist_r,Instructor -0.03657289970847689 t_center_dist_c,CertificateTemplate -0.03657289970847689 t_center_dist_c,Course -0.02264036648619997 t_dist_c,Role,User -0.03657289970847689 t_center_dist_c,CourseSession -0.03657289970847689 t_center_dist_r,Therapist -0.03657289970847689 t_dist_c,Client,Location -0.03657289970847689 t_dist_r,Client,Location -0.05485934956271533 t_center_dist_r,Course -0.03657289970847689 t_center_dist_c,User -0.0950895392420399 t_dist_c,CertificateTemplate,CourseSession -0.025442017188505656 t_dist_c,Client,CertificateTemplate -0.03657289970847689 t_center_dist_r,Client -0.03657289970847689 t_center_dist_r,CertificateTemplate -0.03657289970847689 t_center_dist_r,Role -0.03657289970847689 t_center_dist_c,Client -0.03657289970847689 t_dist_c,Client,User -0.03657289970847689 t_dist_r,Client,CertificateTemplate -0.03657289970847689 t_center_dist_c,Instructor -0.03657289970847689 t_dist_c,Client,CourseSession -0.03657289970847689 t_center_dist_r,User -0.03657289970847689 t_center_dist_c,Therapist -0.03657289970847689 t_dist_r,Client,CourseSession -0.03657289970847689 t_dist_c,Therapist,CourseSession -0.07314579941695377 t_center_dist_r,CourseSession <= -2.1379547949873166
 newcut_12: +0.5188325664825932 t_zV_r,lb,Client,User +0.5188325664825932 t_zV_r,lb,User,Therapist +0.03038880685239282 t_x_c,Location -0.03038880685239282 t_x_c,Clinic -0.0006077761370478555 t_x_c,CourseSession +0.001215552274095711 t_x_c,User +0.0006077761370478555 t_x_c,CertificateTemplate -0.001215552274095711 t_x_c,Client -0.03160435912648853 t_x_r,Therapist +0.030388806852392824 t_x_r,Location +0.002431104548191422 t_x_r,CourseSession -0.001215552274095711 t_x_r,Client -0.0006077761370478555 t_center_dist_r,Clinic -0.0006077761370478555 t_dist_r,Role,User -0.001215552274095711 t_dist_c,User,Therapist -0.0006077761370478555 t_center_dist_c,Location -0.03038880685239282 t_dist_c,Clinic,Location -0.001215552274095711 t_dist_r,Therapist,CourseSession -0.001580217956324424 t_dist_r,CertificateTemplate,CourseSession -0.0006077761370478555 t_center_dist_r,Location -0.0006077761370478555 t_center_dist_c,Clinic -0.03038880685239282 t_dist_c,Clinic,Therapist -0.03038880685239282 t_dist_c,Location,Therapist -0.0006077761370478555 t_center_dist_c,Role -0.0006077761370478555 t_center_dist_r,Instructor -0.0006077761370478555 t_center_dist_c,CertificateTemplate -0.0006077761370478555 t_center_dist_c,Course -0.00037624237055343423 t_dist_c,Role,User -0.0006077761370478555 t_center_dist_c,CourseSession -0.0006077761370478555 t_center_dist_r,Therapist -0.029781030715344967 t_dist_c,Client,Location -0.029781030715344967 t_dist_r,Client,Location -0.03038880685239282 t_dist_r,Location,Therapist -0.0006077761370478555 t_center_dist_r,Course -0.0006077761370478555 t_center_dist_c,User -0.001580217956324424 t_dist_c,CertificateTemplate,CourseSession -0.00042280079098981247 t_dist_c,Client,CertificateTemplate -0.0006077761370478555 t_center_dist_r,Client -0.0006077761370478555 t_center_dist_r,CertificateTemplate -0.0006077761370478555 t_center_dist_r,Role -0.0006077761370478555 t_center_dist_c,Client -0.001215552274095711 t_dist_c,Client,User -0.0006077761370478555 t_dist_r,Client,CertificateTemplate -0.0006077761370478555 t_center_dist_c,Instructor -0.001215552274095711 t_dist_c,Client,CourseSession -0.0006077761370478555 t_center_dist_r,User -0.0006077761370478555 t_center_dist_c,Therapist -0.001215552274095711 t_dist_r,Client,CourseSession -0.001215552274095711 t_dist_c,Therapist,CourseSession -0.002431104548191422 t_center_dist_r,CourseSession <= 0.226613799789984
 newcut_13: +1.0 t_zV_c,lb,Course,Role +1.0 t_zV_c,lb,Role,Therapist -0.02439024390243903 t_x_c,Therapist +0.02439024390243903 t_x_c,Location -0.02439024390243903 t_x_c,Clinic +0.02439024390243903 t_x_c,CourseSession +0.02439024390243903 t_x_c,User +0.02439024390243903 t_x_c,Course -0.02439024390243903 t_x_c,Client -0.02439024390243903 t_x_r,Therapist +0.02439024390243903 t_x_r,Location +0.02439024390243903 t_x_r,CourseSession -0.02439024390243903 t_x_r,Client -0.02439024390243903 t_x_c,Instructor -0.02439024390243903 t_dist_c,User,Therapist -0.02439024390243903 t_dist_c,Clinic,Location -0.02439024390243903 t_dist_r,Therapist,CourseSession -0.02439024390243903 t_dist_c,Clinic,Therapist -0.02439024390243903 t_dist_c,Location,Therapist -0.02439024390243903 t_center_dist_c,Course -0.02439024390243903 t_dist_c,Client,Location -0.02439024390243903 t_dist_r,Client,Location -0.02439024390243903 t_dist_r,Location,Therapist -0.04878048780487806 t_dist_c,CourseSession,Instructor -0.02439024390243903 t_dist_c,Client,User -0.02439024390243903 t_center_dist_c,Instructor -0.02439024390243903 t_dist_c,Client,CourseSession -0.02439024390243903 t_dist_r,Client,CourseSession -0.02439024390243903 t_dist_c,Therapist,CourseSession <= 0.548780487804878
 newcut_14: +0.6667674629573632 t_zV_c,lb,Client,Role +0.6667674629573632 t_zV_c,lb,Role,Location +0.00030238887208950863 t_x_c,Therapist +0.00030238887208951237 t_x_c,Location -0.03235560931357727 t_x_c,Clinic +0.00030238887208950863 t_x_c,CourseSession +0.031750831569398244 t_x_c,User +0.0003023888720895092 t_x_c,Role -0.00030238887208950543 t_x_c,Client -0.00030238887208950863 t_x_r,Therapist +0.00030238887208950863 t_x_r,Location +0.00030238887208950863 t_x_r,CourseSession -0.00030238887208950863 t_x_r,Client -0.00030238887208950863 t_x_c,Instructor -0.031750831569398244 t_dist_c,User,Therapist -0.03235560931357727 t_dist_c,Clinic,Location -0.00030238887208950863 t_dist_r,Therapist,CourseSession -0.03235560931357727 t_dist_c,Clinic,Therapist -0.00030238887208950863 t_dist_c,Location,Therapist -0.00030238887208950863 t_center_dist_c,Role -0.00030238887208950863 t_dist_c,Client,Location -0.00030238887208950863 t_dist_r,Client,Location -0.00030238887208950863 t_dist_r,Location,Therapist -0.0006047777441790173 t_dist_c,CourseSession,Instructor -0.031750831569398244 t_dist_c,Client,User -0.00030238887208950863 t_center_dist_c,Instructor -0.00030238887208950863 t_dist_c,Client,CourseSession -0.00030238887208950863 t_dist_r,Client,CourseSession -0.00030238887208950863 t_dist_c,Therapist,CourseSession <= 0.6611732688237075
 newcut_15: +1.0 t_zV_c,lb,Client,Course +1.0 t_zV_c,lb,Course,Location +0.0004535147392290344 t_x_c,Therapist +0.00045351473922902713 t_x_c,Location -0.04852607709750567 t_x_c,Clinic +0.00045351473922902713 t_x_c,CourseSession +0.04761904761904761 t_x_c,User +0.00045351473922902713 t_x_c,Course -0.00045351473922902713 t_x_c,Client -0.00045351473922902713 t_x_r,Therapist +0.00045351473922902713 t_x_r,Location +0.00045351473922902713 t_x_r,CourseSession -0.00045351473922902713 t_x_r,Client -0.00045351473922902713 t_x_c,Instructor -0.04761904761904761 t_dist_c,User,Therapist -0.04852607709750567 t_dist_c,Clinic,Location -0.00045351473922902713 t_dist_r,Therapist,CourseSession -0.04852607709750567 t_dist_c,Clinic,Therapist -0.00045351473922902713 t_dist_c,Location,Therapist -0.00045351473922902713 t_center_dist_c,Course -0.00045351473922902713 t_dist_c,Client,Location -0.00045351473922902713 t_dist_r,Client,Location -0.00045351473922902713 t_dist_r,Location,Therapist -0.0009070294784580543 t_dist_c,CourseSession,Instructor -0.04761904761904761 t_dist_c,Client,User -0.00045351473922902713 t_center_dist_c,Instructor -0.00045351473922902713 t_dist_c,Client,CourseSession -0.00045351473922902713 t_dist_r,Client,CourseSession -0.00045351473922902713 t_dist_c,Therapist,CourseSession <= 0.991609977324263
 newcut_16: +0.6824445775913723 t_zV_c,lb,CertificateTemplate,Course +0.6824445775913723 t_zV_c,lb,Course,Clinic +0.01617735170760935 t_x_c,Therapist +0.01617735170760935 t_x_c,Location -0.048532055122828044 t_x_c,Clinic -0.014979029358897542 t_x_c,CourseSession +0.01617735170760935 t_x_c,User +0.0005991611743559006 t_x_c,Course +0.031156381066506894 t_x_c,CertificateTemplate -0.01617735170760935 t_x_c,Client -0.01617735170760935 t_x_r,Therapist +0.01617735170760935 t_x_r,Location +0.01617735170760935 t_x_r,CourseSession -0.01617735170760935 t_x_r,Client -0.0005991611743559036 t_x_c,Instructor -0.01617735170760935 t_dist_c,User,Therapist -0.01617735170760935 t_dist_c,Clinic,Location -0.01617735170760935 t_dist_r,Therapist,CourseSession -0.08100659077291793 t_dist_r,CertificateTemplate,CourseSession -0.04853205512282805 t_dist_c,Clinic,Therapist -0.01617735170760935 t_dist_c,Location,Therapist -0.0005991611743559036 t_center_dist_c,Course -0.01617735170760935 t_dist_c,Client,Location -0.01617735170760935 t_dist_r,Client,Location -0.01617735170760935 t_dist_r,Location,Therapist -0.08100659077291793 t_dist_c,CertificateTemplate,CourseSession -0.0011983223487118072 t_dist_c,CourseSession,Instructor -0.01617735170760935 t_dist_c,Client,User -0.0005991611743559036 t_center_dist_c,Instructor -0.01617735170760935 t_dist_c,Client,CourseSession -0.01617735170760935 t_dist_r,Client,CourseSession -0.01617735170760935 t_dist_c,Therapist,CourseSession <= -0.24931096464949049
 newcut_17: +0.6824445775913723 t_zV_c,lb,CertificateTemplate,Role +0.6824445775913723 t_zV_c,lb,Role,Clinic +0.01617735170760935 t_x_c,Therapist +0.01617735170760935 t_x_c,Location -0.048532055122828044 t_x_c,Clinic -0.014979029358897542 t_x_c,CourseSession +0.01617735170760935 t_x_c,User +0.0005991611743559006 t_x_c,Role +0.031156381066506894 t_x_c,CertificateTemplate -0.01617735170760935 t_x_c,Client -0.01617735170760935 t_x_r,Therapist +0.01617735170760935 t_x_r,Location +0.01617735170760935 t_x_r,CourseSession -0.01617735170760935 t_x_r,Client -0.0005991611743559036 t_x_c,Instructor -0.01617735170760935 t_dist_c,User,Therapist -0.01617735170760935 t_dist_c,Clinic,Location -0.01617735170760935 t_dist_r,Therapist,CourseSession -0.08100659077291793 t_dist_r,CertificateTemplate,CourseSession -0.04853205512282805 t_dist_c,Clinic,Therapist -0.01617735170760935 t_dist_c,Location,Therapist -0.0005991611743559036 t_center_dist_c,Role -0.01617735170760935 t_dist_c,Client,Location -0.01617735170760935 t_dist_r,Client,Location -0.01617735170760935 t_dist_r,Location,Therapist -0.08100659077291793 t_dist_c,CertificateTemplate,CourseSession -0.0011983223487118072 t_dist_c,CourseSession,Instructor -0.01617735170760935 t_dist_c,Client,User -0.0005991611743559036 t_center_dist_c,Instructor -0.01617735170760935 t_dist_c,Client,CourseSession -0.01617735170760935 t_dist_r,Client,CourseSession -0.01617735170760935 t_dist_c,Therapist,CourseSession <= -0.24931096464949049
 newcut_18: +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_c,lb,User,Location +0.02201911092646447 t_x_c,Therapist +0.02201911092646447 t_x_c,Location -0.06688824262567508 t_x_c,Clinic -0.02201911092646447 t_x_c,CourseSession +0.02285002077274615 t_x_c,User +0.04403822185292894 t_x_c,CertificateTemplate -0.02201911092646447 t_x_c,Client -0.02201911092646447 t_x_r,Therapist +0.02201911092646447 t_x_r,Location +0.02201911092646447 t_x_r,CourseSession -0.02201911092646447 t_x_r,Client -0.02285002077274615 t_dist_c,User,Therapist -0.06688824262567508 t_dist_c,Clinic,Location -0.02201911092646447 t_dist_r,Therapist,CourseSession -0.11449937681761525 t_dist_r,CertificateTemplate,CourseSession -0.0668882426256751 t_dist_c,Clinic,Therapist -0.02201911092646447 t_dist_c,Location,Therapist -0.02201911092646447 t_dist_c,Client,Location -0.02201911092646447 t_dist_r,Client,Location -0.02201911092646447 t_dist_r,Location,Therapist -0.11449937681761525 t_dist_c,CertificateTemplate,CourseSession -0.02201911092646447 t_dist_c,Client,User -0.02201911092646447 t_dist_c,Client,CourseSession -0.02201911092646447 t_dist_r,Client,CourseSession -0.02201911092646447 t_dist_c,Therapist,CourseSession <= -0.30132945575404996
 newcut_19: +0.6939151813153044 t_zV_c,lb,Client,Course +0.6939151813153046 t_zV_c,lb,Course,Instructor +0.015980331899200985 t_x_c,Location -0.015980331899200985 t_x_c,Clinic +0.015980331899200985 t_x_c,CourseSession +0.015980331899200985 t_x_c,User +0.0003073140749846363 t_x_c,Course -0.015980331899200985 t_x_r,Therapist +0.015980331899200985 t_x_r,Location +0.015980331899200985 t_x_r,CourseSession -0.015980331899200985 t_x_r,Client -0.03226797787338661 t_x_c,Instructor -0.015980331899200985 t_dist_c,User,Therapist -0.015980331899200985 t_dist_c,Clinic,Location -0.015980331899200985 t_dist_r,Therapist,CourseSession -0.015980331899200985 t_dist_c,Clinic,Therapist -0.015980331899200985 t_dist_c,Location,Therapist -0.00030731407498463584 t_center_dist_c,Course -0.015980331899200985 t_dist_c,Client,Location -0.015980331899200985 t_dist_r,Client,Location -0.015980331899200985 t_dist_r,Location,Therapist -0.03196066379840197 t_dist_c,CourseSession,Instructor -0.015980331899200985 t_dist_c,Client,User -0.00030731407498463584 t_center_dist_c,Instructor -0.015980331899200985 t_dist_c,Client,CourseSession -0.015980331899200985 t_dist_r,Client,CourseSession -0.015980331899200985 t_dist_c,Therapist,CourseSession <= 0.3985863552550709
 newcut_20: +0.7101096224116932 t_zV_c,lb,CertificateTemplate,Course +0.7101096224116934 t_zV_c,lb,Course,Instructor +0.0006090133982947631 t_x_c,Course +0.03166869671132765 t_x_c,CertificateTemplate -0.032277710109622416 t_x_c,Instructor -0.0823386114494519 t_dist_r,CertificateTemplate,CourseSession -0.0006090133982947644 t_center_dist_c,Course -0.0823386114494519 t_dist_c,CertificateTemplate,CourseSession -0.03166869671132765 t_dist_c,CourseSession,Instructor -0.0006090133982947644 t_center_dist_c,Instructor <= 0.05200974421437301
 newcut_21: -0.03303598427577882 t_zV_c,lb,Location,Instructor -0.03303598427577882 t_zV_r,lb,Location,Instructor -0.03303598427577882 t_zV_c,rt,Location,Instructor -0.03303598427577882 t_zV_r,rt,Location,Instructor +0.6667674629573632 t_zV_c,lb,Client,Course -0.2557201894970264 t_zV_r,rt,Client,User -0.2557201894970264 t_zV_c,rt,Client,User -0.2557201894970264 t_zV_r,lb,Client,User -0.2557201894970264 t_zV_c,lb,Client,User +0.6667674629573632 t_zV_c,lb,Course,User +0.0006047777441790173 t_x_c,Therapist +0.0006047777441790173 t_x_c,Location -0.0006047777441790173 t_x_c,Clinic -0.032053220441487756 t_x_c,User +0.0003023888720895092 t_x_c,Course +0.03144844269730874 t_x_c,Client -0.00030238887208950863 t_x_c,Instructor -0.002661022074387675 t_dist_r,Clinic,Location -0.0006047777441790173 t_dist_c,User,Therapist -0.002661022074387675 t_dist_c,Clinic,Location -0.11531095655679871 t_dist_r,Client,User -0.0006047777441790173 t_dist_c,Clinic,Therapist -0.00030238887208950863 t_center_dist_c,Course -0.00347747202902935 t_dist_r,Instructor,Location -0.11531095655679871 t_dist_c,Client,User -0.00030238887208950863 t_center_dist_c,Instructor -0.00347747202902935 t_dist_c,Instructor,Location <= -0.5998286463058161
 newcut_22: +1.0 t_zV_c,lb,Client,Role +1.0 t_zV_c,lb,Role,Therapist -0.04761904761904761 t_x_c,Therapist +0.00045351473922902713 t_x_c,Location -0.00045351473922902713 t_x_c,Clinic +0.00045351473922902713 t_x_c,CourseSession +0.04761904761904761 t_x_c,User +0.00045351473922902713 t_x_c,Role -0.00045351473922902713 t_x_c,Client -0.00045351473922902713 t_x_r,Therapist +0.00045351473922902713 t_x_r,Location +0.00045351473922902713 t_x_r,CourseSession -0.00045351473922902713 t_x_r,Client -0.00045351473922902713 t_x_c,Instructor -0.04761904761904761 t_dist_c,User,Therapist -0.00045351473922902713 t_dist_c,Clinic,Location -0.00045351473922902713 t_dist_r,Therapist,CourseSession -0.00045351473922902713 t_dist_c,Clinic,Therapist -0.00045351473922902713 t_dist_c,Location,Therapist -0.00045351473922902713 t_center_dist_c,Role -0.00045351473922902713 t_dist_c,Client,Location -0.00045351473922902713 t_dist_r,Client,Location -0.00045351473922902713 t_dist_r,Location,Therapist -0.0009070294784580543 t_dist_c,CourseSession,Instructor -0.04761904761904761 t_dist_c,Client,User -0.00045351473922902713 t_center_dist_c,Instructor -0.00045351473922902713 t_dist_c,Client,CourseSession -0.00045351473922902713 t_dist_r,Client,CourseSession -0.00045351473922902713 t_dist_c,Therapist,CourseSession <= 0.991609977324263
 newcut_23: +0.6824445775913723 t_zV_c,lb,CertificateTemplate,Role +0.6824445775913723 t_zV_c,lb,Role,Therapist -0.01617735170760935 t_x_c,Therapist +0.01617735170760935 t_x_c,Location -0.01617735170760935 t_x_c,Clinic -0.014979029358897542 t_x_c,CourseSession +0.01617735170760935 t_x_c,User +0.0005991611743559006 t_x_c,Role +0.031156381066506894 t_x_c,CertificateTemplate -0.01617735170760935 t_x_c,Client -0.01617735170760935 t_x_r,Therapist +0.01617735170760935 t_x_r,Location +0.01617735170760935 t_x_r,CourseSession -0.01617735170760935 t_x_r,Client -0.0005991611743559036 t_x_c,Instructor -0.01617735170760935 t_dist_c,User,Therapist -0.01617735170760935 t_dist_c,Clinic,Location -0.01617735170760935 t_dist_r,Therapist,CourseSession -0.08100659077291793 t_dist_r,CertificateTemplate,CourseSession -0.01617735170760935 t_dist_c,Clinic,Therapist -0.01617735170760935 t_dist_c,Location,Therapist -0.0005991611743559036 t_center_dist_c,Role -0.01617735170760935 t_dist_c,Client,Location -0.01617735170760935 t_dist_r,Client,Location -0.01617735170760935 t_dist_r,Location,Therapist -0.08100659077291793 t_dist_c,CertificateTemplate,CourseSession -0.0011983223487118072 t_dist_c,CourseSession,Instructor -0.01617735170760935 t_dist_c,Client,User -0.0005991611743559036 t_center_dist_c,Instructor -0.01617735170760935 t_dist_c,Client,CourseSession -0.01617735170760935 t_dist_r,Client,CourseSession -0.01617735170760935 t_dist_c,Therapist,CourseSession <= -0.24931096464949049
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
