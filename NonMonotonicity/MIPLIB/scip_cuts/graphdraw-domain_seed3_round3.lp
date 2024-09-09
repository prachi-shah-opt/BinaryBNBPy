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
 newcut_1: -0.6642335766423372 t_zV_r,rt,CourseSession,Therapist -0.6642335766423372 t_zV_c,rt,CourseSession,Therapist +1.0 t_zV_r,lb,CourseSession,Therapist -0.6642335766423372 t_zV_c,lb,CourseSession,Therapist -0.0875912408759125 t_x_r,Therapist +0.0875912408759125 t_x_r,CourseSession -0.36496350364963553 t_dist_r,Therapist,CourseSession -0.277372262773723 t_dist_c,Therapist,CourseSession <= -3.255474452554749
 newcut_2: +1.0 t_zV_r,lb,CourseSession,Instructor -0.05263157894736841 t_x_r,Instructor +0.05263157894736841 t_x_r,CourseSession -0.05263157894736841 t_dist_c,CourseSession,Instructor -0.05263157894736841 t_dist_r,CourseSession,Instructor <= 0.026315789473684206
 newcut_3: -1.0 t_zV_r,rt,CourseSession,Instructor -0.05263157894736842 t_x_r,Instructor +0.05263157894736842 t_x_r,CourseSession -0.2591093117408907 t_dist_c,CourseSession,Instructor -0.05263157894736842 t_dist_r,CourseSession,Instructor <= -0.9736842105263159
 newcut_4: +1.0 t_zV_c,lb,Client,Course +1.0 t_zV_c,lb,Course,Role -0.018587360594795627 t_zV_r,rt,Course,CourseSession -0.018587360594795627 t_zV_c,rt,Course,CourseSession -0.018587360594795627 t_zV_r,lb,Course,CourseSession -0.018587360594795627 t_zV_c,lb,Course,CourseSession -0.04925650557620818 t_x_c,Therapist -0.00046468401486989073 t_x_c,CourseSession +0.02462825278810409 t_x_c,User -0.02462825278810409 t_x_c,Role +0.0009293680297397815 t_x_c,Course -0.00046468401486989073 t_x_c,Client -0.04925650557620818 t_dist_c,User,Therapist -0.0024783147459727503 t_dist_c,Course,CourseSession -0.02462825278810409 t_dist_c,Role,User -0.04925650557620818 t_center_dist_c,Client -0.0024783147459727503 t_dist_r,Course,CourseSession -0.00046468401486989073 t_dist_c,Client,CourseSession -0.04925650557620818 t_center_dist_c,Therapist <= 0.11431226765799245
 newcut_5: +1.0 t_zV_c,lb,Client,Role +1.0 t_zV_c,lb,Role,Therapist -0.048072562358276644 t_x_c,Therapist +0.0004535147392290272 t_x_c,User +0.000453514739229028 t_x_c,Role -0.0009070294784580544 t_dist_c,User,Therapist -0.0004535147392290272 t_dist_c,Role,User -0.04716553287981859 t_center_dist_c,Client -0.04716553287981859 t_center_dist_c,Therapist <= 0.19841269841269873
 newcut_6: +1.0 t_zV_c,lb,Client,Therapist +1.0 t_zV_c,lb,Therapist,Instructor +0.02222222222222222 t_x_c,Client -0.02222222222222222 t_x_c,Instructor -0.02222222222222222 t_dist_c,CourseSession,Instructor -0.02222222222222222 t_dist_c,Client,CourseSession <= 1.0111111111111113
 newcut_7: +1.0 t_zV_c,lb,CertificateTemplate,Role +1.0 t_zV_c,lb,Role,Clinic -0.0017559262510974587 t_x_c,Therapist -0.047410008779631245 t_x_c,Clinic -0.022827041264266892 t_x_c,CourseSession +0.0008779631255487293 t_x_c,User +0.0008779631255487299 t_x_c,Role +0.045654082528533785 t_x_c,CertificateTemplate -0.022827041264266892 t_x_c,Client -0.0017559262510974585 t_dist_c,User,Therapist -0.11870061457418785 t_dist_r,CertificateTemplate,CourseSession -0.047410008779631245 t_center_dist_c,Clinic -0.0008779631255487293 t_dist_c,Role,User -0.11870061457418785 t_dist_c,CertificateTemplate,CourseSession -0.045654082528533785 t_center_dist_c,Client -0.022827041264266892 t_dist_c,Client,CourseSession -0.0017559262510974587 t_center_dist_c,Therapist <= -0.7437225636523261
 newcut_8: +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Location -0.023705004389815622 t_x_c,Location -0.023705004389815622 t_x_c,CourseSession +0.0017559262510974541 t_x_c,Course +0.045654082528533785 t_x_c,CertificateTemplate -0.11870061457418785 t_dist_r,CertificateTemplate,CourseSession -0.004682470002926556 t_dist_c,Course,CourseSession -0.11870061457418785 t_dist_c,CertificateTemplate,CourseSession -0.023705004389815622 t_dist_c,CourseSession,Instructor -0.004682470002926556 t_dist_r,Course,CourseSession -0.023705004389815622 t_dist_c,Instructor,Location <= 0.02888498683055342
 newcut_9: +1.0 t_zV_c,lb,Course,User +1.0 t_zV_c,lb,User,Clinic -0.0008861320336730216 t_x_c,Therapist -0.046078865750996886 t_x_c,Clinic -0.023482498892334957 t_x_c,CourseSession +0.046964997784669914 t_x_c,Course -0.023482498892334957 t_x_c,Client -0.0008861320336730216 t_dist_c,User,Therapist -0.1252399940924531 t_dist_c,Course,CourseSession -0.04607886575099689 t_center_dist_c,Clinic -0.046964997784669914 t_center_dist_c,Client -0.1252399940924531 t_dist_r,Course,CourseSession -0.023482498892334957 t_dist_c,Client,CourseSession -0.0008861320336730216 t_center_dist_c,Therapist <= -0.6789986708019491
 newcut_10: +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_c,lb,User,Location -0.0008309098462816789 t_x_c,Therapist -0.02243456584960531 t_x_c,Location -0.02201911092646447 t_x_c,CourseSession +0.0008309098462816799 t_x_c,User +0.04403822185292894 t_x_c,CertificateTemplate -0.00041545492314083943 t_x_c,Client -0.000830909846281679 t_dist_c,User,Therapist -0.11449937681761525 t_dist_r,CertificateTemplate,CourseSession -0.11449937681761525 t_dist_c,CertificateTemplate,CourseSession -0.02243456584960531 t_dist_c,CourseSession,Instructor -0.000830909846281679 t_center_dist_c,Client -0.00041545492314083943 t_dist_c,Client,CourseSession -0.0008309098462816789 t_center_dist_c,Therapist -0.02243456584960531 t_dist_c,Instructor,Location <= 0.08088907353552197
 newcut_11: +1.0 t_zV_c,lb,Role,CourseSession +1.0 t_zV_c,lb,CourseSession,Clinic -0.04236006051437216 t_x_c,Therapist -0.03933434190620272 t_x_c,Clinic -0.0015128593040847171 t_x_c,CourseSession +0.02118003025718608 t_x_c,User +0.02118003025718608 t_x_c,Role -0.0015128593040847223 t_x_c,Client -0.04236006051437216 t_dist_c,User,Therapist -0.03933434190620272 t_center_dist_c,Clinic -0.02118003025718608 t_dist_c,Role,User -0.0030257186081694446 t_center_dist_c,Client -0.0015128593040847223 t_dist_c,Client,CourseSession -0.04236006051437216 t_center_dist_c,Therapist <= 0.2927382753403932
 newcut_12: -1.0 t_zV_r,rt,Client,User +1.0 t_zV_c,lb,Client,Clinic -0.036195883605393886 t_x_c,Clinic -0.037615330021291686 t_x_r,User +0.037615330021291686 t_x_r,Client -0.037615330021291686 t_dist_r,Client,User -0.036195883605393886 t_center_dist_c,Clinic -0.036195883605393886 t_center_dist_c,Client -0.21201367830182588 t_dist_c,Client,User <= -0.6529453513129878
 newcut_13: +1.0 t_zV_r,lb,Role,User -0.07879924953095685 t_x_r,User +0.07879924953095685 t_x_r,Role -0.07879924953095685 t_dist_r,Role,User -0.048780487804878044 t_dist_c,Role,User <= 0.015009380863039405
 newcut_14: +1.0 t_zV_c,lb,User,CourseSession -0.03999999999999999 t_x_c,CourseSession +0.03999999999999999 t_x_c,Instructor -0.3066666666666665 t_dist_r,User,Instructor -0.3066666666666665 t_dist_c,User,Instructor -0.03999999999999999 t_dist_c,CourseSession,Instructor <= -2.566666666666665
 newcut_15: +1.0 t_zV_r,lb,CourseSession,Location +0.06249999999999999 t_x_r,CourseSession -0.06249999999999999 t_x_r,CertificateTemplate -0.06249999999999999 t_dist_r,CertificateTemplate,CourseSession -0.06249999999999999 t_center_dist_r,Location -0.06249999999999999 t_dist_c,CertificateTemplate,CourseSession -0.06249999999999999 t_center_dist_r,CertificateTemplate <= -0.12499999999999999
 newcut_16: +0.45155993431855523 t_zV_r,rt,Location,Instructor +0.451559934318555 t_zV_r,rt,Clinic,Location +0.00054734537493158 t_x_r,Location -0.014230979748221129 t_x_r,Clinic +0.027914614121510677 t_x_r,CourseSession -0.027914614121510677 t_x_r,CertificateTemplate -0.014230979748221129 t_center_dist_r,Clinic -0.027914614121510677 t_dist_r,CertificateTemplate,CourseSession -0.0005473453749315799 t_center_dist_r,Location -0.027914614121510677 t_dist_c,CertificateTemplate,CourseSession -0.027914614121510677 t_dist_c,CourseSession,Instructor -0.027914614121510677 t_center_dist_r,CertificateTemplate -0.027914614121510677 t_dist_r,CourseSession,Instructor <= -0.10207991242473992
 newcut_17: -0.010821990284257624 t_zV_r,rt,Client,CourseSession -0.010821990284257623 t_zV_c,rt,Client,CourseSession -0.010821990284257624 t_zV_r,lb,Client,CourseSession -0.010821990284257623 t_zV_c,lb,Client,CourseSession +1.0 t_zV_r,lb,Course,User -0.03917961297356232 t_zV_r,rt,User,Instructor -0.03917961297356232 t_zV_c,rt,User,Instructor +0.9608203870264377 t_zV_r,lb,User,Instructor -0.03917961297356232 t_zV_c,lb,User,Instructor -0.0012024433649175136 t_x_c,CourseSession -0.0012024433649175136 t_x_c,User +0.0012024433649175136 t_x_c,CertificateTemplate -0.057717281516040594 t_x_r,Instructor +0.0589197248809581 t_x_r,CourseSession -0.0012024433649175084 t_x_r,User +0.0012024433649175136 t_x_c,Instructor -0.004609366232183802 t_dist_r,User,Instructor -0.0012024433649175136 t_dist_r,Client,User -0.058919724880958105 t_dist_c,Course,CourseSession -0.004609366232183802 t_dist_c,User,Instructor -0.0012024433649175136 t_dist_c,CertificateTemplate,CourseSession -0.057717281516040594 t_dist_c,CourseSession,Instructor -0.0012024433649175138 t_dist_c,Client,CertificateTemplate -0.0012024433649175136 t_dist_c,Client,User -0.0589197248809581 t_dist_r,Course,CourseSession -0.0008657592227406099 t_dist_c,Client,CourseSession -0.057717281516040594 t_dist_r,CourseSession,Instructor -0.0012024433649175136 t_dist_r,Client,CourseSession <= 0.4967453866256232
 newcut_18: +0.5100097934270165 t_zV_r,rt,Clinic,Instructor +0.5100097934270162 t_zV_r,lb,Client,Instructor -0.0005693852916391455 t_x_r,Instructor -0.014519324936798235 t_x_r,Clinic +0.029608035165235615 t_x_r,CourseSession -0.02903864987359647 t_x_r,CertificateTemplate -0.014519324936798235 t_center_dist_r,Clinic -0.02903864987359647 t_dist_r,CertificateTemplate,CourseSession -0.02903864987359647 t_dist_c,CertificateTemplate,CourseSession -0.0005693852916391444 t_dist_c,CourseSession,Instructor -0.02903864987359647 t_center_dist_r,CertificateTemplate -0.02131778531896964 t_dist_c,Client,CourseSession -0.0005693852916391444 t_dist_r,CourseSession,Instructor -0.029608035165235615 t_dist_r,Client,CourseSession <= -0.061083654087047426
 newcut_19: +1.0 t_zV_c,lb,Client,Course -0.018140589569161085 t_zV_r,rt,Course,CourseSession -0.018140589569161085 t_zV_c,rt,Course,CourseSession -0.018140589569161085 t_zV_r,lb,Course,CourseSession -0.018140589569161085 t_zV_c,lb,Course,CourseSession +1.0 t_zV_c,lb,Course,Therapist -0.048072562358276644 t_x_c,Therapist -0.00045351473922902713 t_x_c,CourseSession +0.0009070294784580543 t_x_c,Course -0.00045351473922902713 t_x_c,Client -0.0024187452758881447 t_dist_c,Course,CourseSession -0.048072562358276644 t_center_dist_c,Client -0.0024187452758881447 t_dist_r,Course,CourseSession -0.00045351473922902713 t_dist_c,Client,CourseSession -0.048072562358276644 t_center_dist_c,Therapist <= 0.1476190476190475
 newcut_20: +1.0 t_zV_c,lb,CertificateTemplate,Course +1.0 t_zV_c,lb,Course,Instructor -0.023156089193825037 t_x_c,CourseSession +0.0017152658662092652 t_x_c,Course +0.04459691252144081 t_x_c,CertificateTemplate -0.023156089193825037 t_x_c,Instructor -0.001715265866209267 t_dist_c,Course,CourseSession -0.04459691252144081 t_dist_c,CertificateTemplate,CourseSession -0.023156089193825037 t_dist_c,CourseSession,Instructor <= 1.0248713550600344
 newcut_21: +1.0 t_zV_c,lb,Client,Role +1.0 t_zV_c,lb,Role,Instructor -0.02887313543051261 t_zV_r,rt,User,Instructor -0.028873135430512606 t_zV_c,rt,User,Instructor -0.02887313543051261 t_zV_r,lb,User,Instructor -0.028873135430512606 t_zV_c,lb,User,Instructor -0.023039432875498443 t_x_c,CourseSession +0.046078865750996886 t_x_c,CertificateTemplate -0.023039432875498447 t_x_c,Instructor -0.003396839462413248 t_dist_r,User,Instructor -0.0008861320336730215 t_dist_c,Role,User -0.0033968394624132483 t_dist_c,User,Instructor -0.04607886575099689 t_dist_c,CertificateTemplate,CourseSession -0.023039432875498443 t_dist_c,CourseSession,Instructor -0.046078865750996886 t_dist_c,Client,CertificateTemplate <= 0.9648500959976368
 newcut_22: +1.0 t_zV_c,lb,CertificateTemplate,Role +1.0 t_zV_c,lb,Role,Location -0.0017559262510974587 t_x_c,Therapist -0.023705004389815622 t_x_c,Location -0.022827041264266892 t_x_c,CourseSession +0.0008779631255487293 t_x_c,User +0.0008779631255487249 t_x_c,Role +0.045654082528533785 t_x_c,CertificateTemplate -0.0008779631255487293 t_x_c,Client -0.0017559262510974585 t_dist_c,User,Therapist -0.11870061457418785 t_dist_r,CertificateTemplate,CourseSession -0.0008779631255487293 t_dist_c,Role,User -0.11870061457418785 t_dist_c,CertificateTemplate,CourseSession -0.023705004389815622 t_dist_c,CourseSession,Instructor -0.0017559262510974585 t_center_dist_c,Client -0.0008779631255487293 t_dist_c,Client,CourseSession -0.0017559262510974587 t_center_dist_c,Therapist -0.023705004389815622 t_dist_c,Instructor,Location <= 0.03239683933274834
 newcut_23: -1.0 t_zV_r,lb,Location,Instructor -0.6639899623588458 t_zV_r,rt,CertificateTemplate,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Location -0.01355081555834379 t_x_c,Therapist -0.0521957340025094 t_x_c,Location -0.07352572145545798 t_x_c,CourseSession +0.03864491844416562 t_x_c,CertificateTemplate -0.006775407779171898 t_x_c,Client -0.01355081555834379 t_x_r,Therapist +0.05420326223337516 t_x_r,CourseSession -0.02710163111668758 t_x_r,User -0.013550815558343792 t_x_r,Client +0.03964868255959849 t_x_c,Instructor -0.01355081555834379 t_center_dist_r,Clinic -0.01355081555834379 t_dist_r,Role,User -0.01355081555834379 t_dist_c,User,Therapist -0.01355081555834379 t_center_dist_c,Location -0.05646173149309914 t_dist_r,Therapist,CourseSession -0.03232120451693856 t_dist_r,CertificateTemplate,CourseSession -0.05420326223337516 t_center_dist_r,Location -0.02710163111668758 t_dist_r,Client,User -0.01355081555834379 t_dist_c,Course,CourseSession -0.01355081555834379 t_center_dist_c,Clinic -0.01355081555834379 t_center_dist_c,Role -0.01355081555834379 t_center_dist_r,Instructor -0.01355081555834379 t_center_dist_c,CertificateTemplate -0.01355081555834379 t_center_dist_c,Course -0.01355081555834379 t_dist_c,Role,User -0.01355081555834379 t_center_dist_c,CourseSession -0.01355081555834379 t_center_dist_r,Therapist -0.01355081555834379 t_center_dist_r,Course -0.01355081555834379 t_center_dist_c,User -0.07096612296110419 t_dist_c,CertificateTemplate,CourseSession -0.053199498117942286 t_dist_c,CourseSession,Instructor -0.01355081555834379 t_center_dist_r,Client -0.01355081555834379 t_center_dist_r,CertificateTemplate -0.01355081555834379 t_center_dist_r,Role -0.06775407779171896 t_center_dist_c,Client -0.02710163111668758 t_dist_c,Client,User -0.01355081555834379 t_center_dist_c,Instructor -0.01355081555834379 t_dist_r,Course,CourseSession -0.04742785445420327 t_dist_c,Client,CourseSession -0.01355081555834379 t_center_dist_r,User -0.02710163111668758 t_center_dist_c,Therapist -0.24509301183787027 t_dist_c,Instructor,Location -0.04065244667503137 t_dist_r,CourseSession,Instructor -0.04065244667503137 t_dist_r,Client,CourseSession -0.04291091593475535 t_dist_c,Therapist,CourseSession -0.01355081555834379 t_center_dist_r,CourseSession <= -2.7361104140526984
 newcut_24: +0.5219512195121951 t_zV_r,rt,Clinic,Instructor +0.5219512195121951 t_zV_r,lb,Course,Instructor -0.002439024390243901 t_x_r,Instructor -0.014634146341463415 t_x_r,Clinic +0.002439024390243899 t_x_r,CourseSession +0.014634146341463417 t_x_r,Course -0.014634146341463415 t_center_dist_r,Clinic -0.002439024390243899 t_dist_c,Course,CourseSession -0.014634146341463415 t_center_dist_r,Course -0.002439024390243899 t_dist_c,CourseSession,Instructor -0.002439024390243899 t_dist_r,Course,CourseSession -0.002439024390243899 t_dist_r,CourseSession,Instructor <= 0.5195121951219513
 newcut_25: +1.0 t_zV_c,lb,Client,Course -0.018140589569161085 t_zV_r,rt,Course,CourseSession -0.018140589569161085 t_zV_c,rt,Course,CourseSession -0.018140589569161085 t_zV_r,lb,Course,CourseSession -0.018140589569161085 t_zV_c,lb,Course,CourseSession +1.0 t_zV_c,lb,Course,Clinic -0.048072562358276644 t_x_c,Clinic -0.00045351473922902713 t_x_c,CourseSession +0.0009070294784580543 t_x_c,Course -0.00045351473922902713 t_x_c,Client -0.0024187452758881447 t_dist_c,Course,CourseSession -0.048072562358276644 t_center_dist_c,Clinic -0.048072562358276644 t_center_dist_c,Client -0.0024187452758881447 t_dist_r,Course,CourseSession -0.00045351473922902713 t_dist_c,Client,CourseSession <= 0.1476190476190475
 newcut_26: +1.0 t_zV_c,lb,Client,Course +1.0 t_zV_c,lb,Course,User -0.048072562358276644 t_x_c,Therapist +0.0009070294784580552 t_x_c,Course +0.04716553287981859 t_x_c,CertificateTemplate -0.048072562358276644 t_dist_c,User,Therapist -0.0009070294784580544 t_dist_c,Course,CourseSession -0.04716553287981859 t_dist_c,CertificateTemplate,CourseSession -0.04716553287981859 t_dist_c,Client,CertificateTemplate -0.048072562358276644 t_dist_c,Therapist,CourseSession <= 0.97687074829932
 newcut_27: +1.0 t_zV_r,lb,Course,User -0.014279624893435195 t_zV_r,rt,User,Instructor -0.014279624893435195 t_zV_c,rt,User,Instructor -0.014279624893435195 t_zV_r,lb,User,Instructor -0.014279624893435195 t_zV_c,lb,User,Instructor -0.0319693094629156 t_x_c,CourseSession -0.0639386189258312 t_x_c,User +0.0639386189258312 t_x_c,CertificateTemplate +0.0639386189258312 t_x_r,CourseSession -0.0639386189258312 t_x_r,User +0.0319693094629156 t_x_c,Instructor -0.2450980392156862 t_dist_r,User,Instructor -0.0639386189258312 t_dist_r,Client,User -0.0639386189258312 t_dist_c,Course,CourseSession -0.2450980392156862 t_dist_c,User,Instructor -0.0639386189258312 t_dist_c,CertificateTemplate,CourseSession -0.0319693094629156 t_dist_c,CourseSession,Instructor -0.0639386189258312 t_dist_c,Client,CertificateTemplate -0.0639386189258312 t_dist_c,Client,User -0.0639386189258312 t_dist_r,Course,CourseSession -0.04603580562659847 t_dist_c,Client,CourseSession -0.0639386189258312 t_dist_r,Client,CourseSession <= -2.6641091219096324
 newcut_28: -0.7630232046974592 t_zV_r,rt,Client,CourseSession -0.7630232046974591 t_zV_c,rt,Client,CourseSession -0.7630232046974592 t_zV_r,lb,Client,CourseSession -0.7630232046974591 t_zV_c,lb,Client,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Therapist -1.0 t_zV_r,rt,User,Therapist -0.16171067918485243 t_x_c,Therapist +0.16171067918485243 t_x_c,CertificateTemplate -0.08478035607749546 t_x_r,Therapist +0.08478035607749546 t_x_r,CourseSession -0.462438305877248 t_dist_c,User,Therapist -0.08478035607749546 t_dist_r,Therapist,CourseSession -0.08478035607749546 t_dist_r,Client,User -0.16171067918485243 t_dist_c,CertificateTemplate,CourseSession -0.08478035607749546 t_dist_c,Client,CertificateTemplate -0.08478035607749546 t_dist_c,Client,User -0.06104185637579673 t_dist_c,Client,CourseSession -0.08478035607749546 t_dist_r,Client,CourseSession -0.16171067918485243 t_dist_c,Therapist,CourseSession <= -2.2427544195685605
 newcut_29: -1.0 t_zV_r,rt,CertificateTemplate,CourseSession +1.0 t_zV_c,lb,CertificateTemplate,Instructor -0.31773958164423216 t_zV_r,rt,CourseSession,Location -0.31773958164423216 t_zV_c,rt,CourseSession,Location -0.31773958164423216 t_zV_r,lb,CourseSession,Location -0.31773958164423216 t_zV_c,lb,CourseSession,Location -1.0 t_zV_r,rt,CourseSession,Instructor +0.07879303821415086 t_x_c,CertificateTemplate -0.22393810860863983 t_x_r,Instructor +0.027241770715096637 t_x_r,Location +0.22393810860863983 t_x_r,CourseSession -0.013620885357548414 t_x_r,Course -0.07879303821415118 t_x_c,Instructor -0.3969515161342653 t_dist_r,Location,CourseSession -0.08683314415437052 t_dist_r,CertificateTemplate,CourseSession -0.08683314415437052 t_center_dist_r,Location -0.027241770715096886 t_dist_c,Course,CourseSession -0.3969515161342653 t_dist_c,Location,CourseSession -0.013620885357548414 t_center_dist_r,Course -0.4141273028900748 t_dist_c,CertificateTemplate,CourseSession -1.1024645346886883 t_dist_c,CourseSession,Instructor -0.08683314415437052 t_center_dist_r,CertificateTemplate -0.027241770715096886 t_dist_r,Course,CourseSession -0.22393810860863983 t_dist_r,CourseSession,Instructor <= -5.532590699222542
 newcut_30: +1.0 t_zV_c,lb,Client,Course +1.0 t_zV_c,lb,Course,Instructor -0.023472099202834357 t_x_c,CourseSession +0.000885739592559792 t_x_c,Course +0.046058458813108924 t_x_c,CertificateTemplate -0.023472099202834357 t_x_c,Instructor -0.0008857395925597914 t_dist_c,Course,CourseSession -0.046058458813108924 t_dist_c,CertificateTemplate,CourseSession -0.023472099202834357 t_dist_c,CourseSession,Instructor -0.046058458813108924 t_dist_c,Client,CertificateTemplate <= 1.024357838795394
 newcut_31: +1.0 t_zV_c,lb,CertificateTemplate,User +1.0 t_zV_c,lb,User,Clinic -0.0008309098462816789 t_x_c,Therapist -0.04486913169921062 t_x_c,Clinic -0.02201911092646447 t_x_c,CourseSession +0.0008309098462816799 t_x_c,User +0.04403822185292894 t_x_c,CertificateTemplate -0.02201911092646447 t_x_c,Client -0.0008309098462816789 t_dist_c,User,Therapist -0.11449937681761525 t_dist_r,CertificateTemplate,CourseSession -0.04486913169921062 t_center_dist_c,Clinic -0.11449937681761525 t_dist_c,CertificateTemplate,CourseSession -0.04403822185292894 t_center_dist_c,Client -0.02201911092646447 t_dist_c,Client,CourseSession -0.0008309098462816789 t_center_dist_c,Therapist <= -0.6677606979642701
 newcut_32: +1.0 t_zV_c,lb,CertificateTemplate,Role +1.0 t_zV_c,lb,Role,Instructor -0.05593705293276124 t_zV_r,rt,User,Instructor -0.05593705293276124 t_zV_c,rt,User,Instructor -0.05593705293276124 t_zV_r,lb,User,Instructor -0.05593705293276124 t_zV_c,lb,User,Instructor -0.0223175965665236 t_x_c,CourseSession +0.0446351931330472 t_x_c,CertificateTemplate -0.0223175965665236 t_x_c,Instructor -0.0065808297567954395 t_dist_r,User,Instructor -0.0017167381974248976 t_dist_c,Role,User -0.00658082975679544 t_dist_c,User,Instructor -0.04463519313304721 t_dist_c,CertificateTemplate,CourseSession -0.0223175965665236 t_dist_c,CourseSession,Instructor <= 0.9095851216022891
 newcut_33: +0.5 t_x_c,Therapist -0.5 t_x_c,CourseSession -1.0 t_dist_c,Therapist,CourseSession <= 0.5
 newcut_34: +0.5 t_x_c,Client -0.5 t_x_c,Instructor -1.0 t_dist_c,Client,Instructor <= 0.0
 newcut_35: +0.5462490805041968 t_zV_r,rt,Clinic,Therapist +1.5462490805041966 t_zV_c,lb,Client,Therapist +1.0 t_zV_c,lb,Therapist,Instructor -0.02917451095290937 t_x_c,Therapist +0.029174510952909394 t_x_c,CourseSession +0.005186579724961647 t_x_c,CertificateTemplate +0.02917451095290937 t_x_c,Client -0.014867779620232665 t_x_r,Clinic +0.029735559240465326 t_x_r,CourseSession -0.014867779620232668 t_x_r,Course -0.03436109067787104 t_x_c,Instructor -0.014867779620232658 t_center_dist_r,Clinic -0.029735559240465326 t_dist_r,Therapist,CourseSession -0.029735559240465344 t_dist_c,Course,CourseSession -0.014867779620232668 t_center_dist_r,Course -0.005186579724961647 t_dist_c,CertificateTemplate,CourseSession -0.03436109067787104 t_dist_c,CourseSession,Instructor -0.0051865797249616525 t_dist_c,Client,CertificateTemplate -0.029735559240465344 t_dist_r,Course,CourseSession -0.034361090677871034 t_dist_c,Client,CourseSession -0.0589100701933747 t_dist_c,Therapist,CourseSession <= 1.2048511975264018
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
