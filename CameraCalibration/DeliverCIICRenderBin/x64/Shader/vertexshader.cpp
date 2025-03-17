varying vec4 vColor;
varying vec3 N;
varying vec3 v;

attribute vec3 tangent; 
attribute vec3 binormal; 
varying mat3 TBNMatrix;

void main(void)
{
   vColor = gl_Color;
   v = vec3(gl_ModelViewMatrix * gl_Vertex);       
   N = normalize(gl_NormalMatrix * gl_Normal);
   
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   
   gl_TexCoord[0] = gl_MultiTexCoord0;
   gl_TexCoord[1] = gl_MultiTexCoord1;
   
   vec3 t;
   t = gl_NormalMatrix*vec3(1,0,0);
   vec3 b = normalize(cross(N,t));
   t = normalize(cross(b,N));
 
   TBNMatrix =   mat3(t, b, N) ; 
   //lightvec = gl_LightSource[0].position.xyz - gl_Vertex.xyz; 
   //lightvec *= TBNMatrix;
   //lightvec = normalize(lightvec);
   //position = gl_Vertex.xyz;   
}