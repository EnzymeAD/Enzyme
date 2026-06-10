; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=0 -passes="enzyme,function(sroa,mem2reg,early-cse,%simplifycfg,instsimplify,correlated-propagation,%simplifycfg,adce)" -S -enzyme-detect-readthrow=0 | FileCheck %s; fi

; Reproducer for an outer guard packing inactive branch values into the tape.
; The inactive tape entries must be defined values, not undef, otherwise SROA
; exposes undef-fed PHIs in augmented_eval.

%State = type { i1, i1, i1, ptr, [3 x double], [3 x double], [3 x double], double }

declare double @__enzyme_autodiff(i8*, ...)
declare void @llvm.memset.p0.i64(ptr writeonly, i8, i64, i1 immarg)
declare double @llvm.fabs.f64(double)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.sqrt.f64(double)

define internal double @dot3(ptr %lhs, ptr %rhs) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %lhs.ptr = getelementptr inbounds double, ptr %lhs, i64 %i
  %rhs.ptr = getelementptr inbounds double, ptr %rhs, i64 %i
  %lhs.val = load double, ptr %lhs.ptr, align 8
  %rhs.val = load double, ptr %rhs.ptr, align 8
  %mul = fmul fast double %lhs.val, %rhs.val
  %sum = fadd fast double %acc, %mul
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 3
  br i1 %done, label %exit, label %loop

exit:
  ret double %sum
}

define internal void @frame_vel(
    ptr %face_centroid,
    ptr %velocity_translational,
    ptr %omega_rot,
    ptr %axis_origin,
    i1 %has_rotation,
    i1 %has_translation,
    ptr %velocity_frame) {
entry:
  %out1 = getelementptr inbounds double, ptr %velocity_frame, i64 1
  %out2 = getelementptr inbounds double, ptr %velocity_frame, i64 2
  call void @llvm.memset.p0.i64(ptr %velocity_frame, i8 0, i64 24, i1 false)
  br i1 %has_rotation, label %rot, label %post.rot

rot:
  %face0 = load double, ptr %face_centroid, align 8
  %origin0 = load double, ptr %axis_origin, align 8
  %rel0 = fsub fast double %face0, %origin0
  %face1.ptr = getelementptr inbounds double, ptr %face_centroid, i64 1
  %origin1.ptr = getelementptr inbounds double, ptr %axis_origin, i64 1
  %face1 = load double, ptr %face1.ptr, align 8
  %origin1 = load double, ptr %origin1.ptr, align 8
  %rel1 = fsub fast double %face1, %origin1
  %face2.ptr = getelementptr inbounds double, ptr %face_centroid, i64 2
  %origin2.ptr = getelementptr inbounds double, ptr %axis_origin, i64 2
  %face2 = load double, ptr %face2.ptr, align 8
  %origin2 = load double, ptr %origin2.ptr, align 8
  %rel2 = fsub fast double %face2, %origin2

  %omega1.ptr = getelementptr inbounds double, ptr %omega_rot, i64 1
  %omega2.ptr = getelementptr inbounds double, ptr %omega_rot, i64 2
  %omega1 = load double, ptr %omega1.ptr, align 8
  %omega2 = load double, ptr %omega2.ptr, align 8
  %mul0a = fmul fast double %omega1, %rel2
  %mul0b = fmul fast double %omega2, %rel1
  %vf0 = fsub fast double %mul0a, %mul0b
  store double %vf0, ptr %velocity_frame, align 8

  %omega0 = load double, ptr %omega_rot, align 8
  %mul1a = fmul fast double %omega2, %rel0
  %mul1b = fmul fast double %omega0, %rel2
  %vf1 = fsub fast double %mul1a, %mul1b
  store double %vf1, ptr %out1, align 8

  %mul2a = fmul fast double %omega0, %rel1
  %mul2b = fmul fast double %omega1, %rel0
  %vf2 = fsub fast double %mul2a, %mul2b
  store double %vf2, ptr %out2, align 8
  br label %post.rot

post.rot:
  %rot.out2 = phi double [ %vf2, %rot ], [ 0.000000e+00, %entry ]
  %rot.out1 = phi double [ %vf1, %rot ], [ 0.000000e+00, %entry ]
  %rot.out0 = phi double [ %vf0, %rot ], [ 0.000000e+00, %entry ]
  br i1 %has_translation, label %trans, label %exit

trans:
  %trans0 = load double, ptr %velocity_translational, align 8
  %sum0 = fadd fast double %rot.out0, %trans0
  store double %sum0, ptr %velocity_frame, align 8
  %trans1.ptr = getelementptr inbounds double, ptr %velocity_translational, i64 1
  %trans1 = load double, ptr %trans1.ptr, align 8
  %sum1 = fadd fast double %rot.out1, %trans1
  store double %sum1, ptr %out1, align 8
  %trans2.ptr = getelementptr inbounds double, ptr %velocity_translational, i64 2
  %trans2 = load double, ptr %trans2.ptr, align 8
  %sum2 = fadd fast double %rot.out2, %trans2
  store double %sum2, ptr %out2, align 8
  br label %exit

exit:
  ret void
}

define internal double @eval(ptr %state, ptr %face_centroid, ptr %area_weighted_normal, double %area) {
entry:
  %velocity_frame = alloca [3 x double], align 16
  %velocity_frame.ptr = getelementptr inbounds [3 x double], ptr %velocity_frame, i64 0, i64 0
  call void @llvm.memset.p0.i64(ptr %velocity_frame.ptr, i8 0, i64 24, i1 false)

  %has_frame_motion.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 0
  %has_frame_motion = load i1, ptr %has_frame_motion.ptr, align 1
  br i1 %has_frame_motion, label %with.frame, label %after.frame

with.frame:
  %has_rotation.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 1
  %has_rotation = load i1, ptr %has_rotation.ptr, align 1
  %has_translation.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 2
  %has_translation = load i1, ptr %has_translation.ptr, align 1
  %omega_rot.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 4, i32 0
  %velocity_translational.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 5, i32 0
  %axis_origin.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 6, i32 0
  call void @frame_vel(ptr %face_centroid, ptr %velocity_translational.ptr, ptr %omega_rot.ptr, ptr %axis_origin.ptr, i1 %has_rotation, i1 %has_translation, ptr %velocity_frame.ptr)
  br label %after.frame

after.frame:
  br i1 %has_frame_motion, label %with.dot, label %after.dot

with.dot:
  %dot = call double @dot3(ptr %velocity_frame.ptr, ptr %area_weighted_normal)
  %frame_velocity_dot_n = fdiv fast double %dot, %area
  br label %after.dot

after.dot:
  %frame_velocity_dot_n.merge = phi double [ %frame_velocity_dot_n, %with.dot ], [ 0.000000e+00, %after.frame ]

  %axis_origin0 = getelementptr inbounds %State, ptr %state, i32 0, i32 6, i32 0
  %origin_x = load double, ptr %axis_origin0, align 8
  %face_x = load double, ptr %face_centroid, align 8
  %dx = fsub fast double %face_x, %origin_x
  %abs_dx = call double @llvm.fabs.f64(double %dx)
  %wall_distance_base = fadd fast double %abs_dx, 5.000000e-01

  %face_y.ptr = getelementptr inbounds double, ptr %face_centroid, i64 1
  %face_y = load double, ptr %face_y.ptr, align 8
  %abs_y = call double @llvm.fabs.f64(double %face_y)
  %umag_base = fadd fast double %abs_y, 1.000000e+00

  %second_point_indices.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 3
  %second_point_indices = load ptr, ptr %second_point_indices.ptr, align 8
  %has_second_point_ptr = icmp ne ptr %second_point_indices, null
  br i1 %has_second_point_ptr, label %check.second, label %no.second

check.second:
  %second_index = load i32, ptr %second_point_indices, align 4
  %second_point_found = icmp sgt i32 %second_index, -1
  br i1 %second_point_found, label %have.second, label %no.second

have.second:
  %next_index = add nuw nsw i32 %second_index, 1
  %next_index_d = uitofp i32 %next_index to double
  %extra_distance = fmul fast double %next_index_d, 2.500000e-01
  %wall_distance_plus = fadd fast double %wall_distance_base, %extra_distance
  %umag_plus = fadd fast double %abs_y, 1.500000e+00
  br label %after.second

no.second:
  %roughness_limit.ptr = getelementptr inbounds %State, ptr %state, i32 0, i32 7
  %roughness_limit = load double, ptr %roughness_limit.ptr, align 8
  %wall_distance_cap = call double @llvm.minnum.f64(double %wall_distance_base, double %roughness_limit)
  br label %after.second

after.second:
  %wall_distance = phi double [ %wall_distance_plus, %have.second ], [ %wall_distance_cap, %no.second ]
  %umag = phi double [ %umag_plus, %have.second ], [ %umag_base, %no.second ]
  %sum0 = fadd fast double %wall_distance, %frame_velocity_dot_n.merge
  %sum1 = fadd fast double %sum0, %umag
  %result = fmul fast double %sum1, 1.500000e+00
  ret double %result
}

define internal void @kernel(ptr %state, ptr %face_centroid, ptr %area_weighted_normal, ptr %output) {
entry:
  %area_sq = call double @dot3(ptr %area_weighted_normal, ptr %area_weighted_normal)
  %area = call double @llvm.sqrt.f64(double %area_sq)
  %value = call double @eval(ptr %state, ptr %face_centroid, ptr %area_weighted_normal, double %area)
  %output0 = load double, ptr %output, align 8
  %updated = fadd fast double %output0, %value
  store double %updated, ptr %output, align 8
  ret void
}

define void @caller(ptr %state, ptr %face_centroid, ptr %dface_centroid, ptr %area_weighted_normal, ptr %output, ptr %doutput) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(
      i8* bitcast (void (ptr, ptr, ptr, ptr)* @kernel to i8*),
      metadata !"enzyme_const", ptr %state,
      metadata !"enzyme_dup", ptr %face_centroid, ptr %dface_centroid,
      metadata !"enzyme_const", ptr %area_weighted_normal,
      metadata !"enzyme_dup", ptr %output, ptr %doutput)
  ret void
}

; CHECK: define internal void @diffekernel(
; CHECK: define internal { ptr, double } @augmented_dot3(
; CHECK-LABEL: define internal { {{.*}}, double } @augmented_eval(
; CHECK: after.frame:
; CHECK-DAG: %[[TRANSBOOL:[^ ]+]] = phi i1 [ %has_translation, %with.frame ], [ false, %entry ]
; CHECK-DAG: %[[ROTBOOL:[^ ]+]] = phi i1 [ %has_rotation, %with.frame ], [ false, %entry ]
; CHECK-DAG: %[[AUG2:[^ ]+]] = phi double [ %_augmented.fca.2.extract, %with.frame ], [ 0.000000e+00, %entry ]
; CHECK-DAG: %[[AUG1:[^ ]+]] = phi double [ %_augmented.fca.1.extract, %with.frame ], [ 0.000000e+00, %entry ]
; CHECK-DAG: %[[AUG0:[^ ]+]] = phi double [ %_augmented.fca.0.extract, %with.frame ], [ 0.000000e+00, %entry ]
; CHECK: %[[DOTPTR:[^ ]+]] = phi ptr [ %subcache, %with.dot ], [ null, %after.frame ]
; CHECK: %[[JOINDOUBLE:[^ ]+]] = phi double [ 0.000000e+00, %have.second ], [ %roughness_limit, %no.second ]
; CHECK: insertvalue {{.*}} double %[[AUG0]], 0, 2, 0
; CHECK: insertvalue {{.*}} double %[[AUG1]], 0, 2, 1
; CHECK: insertvalue {{.*}} double %[[AUG2]], 0, 2, 2
; CHECK: insertvalue {{.*}} ptr %[[DOTPTR]], 0, 3
; CHECK: insertvalue {{.*}} i1 %[[ROTBOOL]], 0, 5
; CHECK: insertvalue {{.*}} i1 %[[TRANSBOOL]], 0, 6
; CHECK: insertvalue {{.*}} double %[[JOINDOUBLE]], 0, 11
