!mod$ v1 sum:13f4c9ccbb9a2e46
module m
type::t
real(8)::a
integer(4)::n
end type
contains
subroutine scalars(x8,x4,i,l,c,z)
real(8)::x8
real(4)::x4
integer(4)::i
logical(4)::l
character(*,1)::c
complex(8)::z
end
subroutine arrays(n,a,b,s)
integer(4)::n
real(8)::a(1_8:__builtin_int(n,kind=8))
real(8)::b(1_8:*)
real(4)::s(1_8:10_8,1_8:3_8)
end
subroutine descr(a,p,q,u)
real(8)::a(:)
real(8),allocatable::p(:,:)
real(4),pointer::q(:)
class(*)::u
end
subroutine byvalue(x,n,tt,r)
real(8),value::x
integer(4),value::n
type(t)::tt
integer(4)::r(..)
end
function f(x)
real(8)::x
real(8)::f
end
end
